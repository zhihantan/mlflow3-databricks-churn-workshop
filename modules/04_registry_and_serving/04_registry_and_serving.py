# Databricks notebook source
# MAGIC %md
# MAGIC # Module 04 — UC Model Registry + Model Serving
# MAGIC ### Promote a model to production — UC-governed, alias-routed, REST-callable
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > **TL;DR** — Register the tuned LightGBM model in Unity Catalog under a three-part name, set `@champion` / `@challenger` aliases, kick off a Model Serving endpoint in the first cell (provisions in background ~5-7 min), and absorb the wait with productive batch-scoring work via `models:/<name>@champion`. By the end you have a REST endpoint your renewal pipeline can call at decision time.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The runtime-critical module of the classic-ML track. Model Serving endpoint provisioning takes 5-10 minutes, so this notebook is structured to **kick the endpoint off early and absorb the wait with productive work** (registration, aliasing, batch scoring) — the *background-provisioning pattern* you'll see again in Module 8.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Register the Module 3 tuned LightGBM `LoggedModel` in Unity Catalog under a three-part name.
# MAGIC - Set `@champion` and `@challenger` aliases on different model versions.
# MAGIC - Batch-score the customer snapshot via `mlflow.pyfunc.spark_udf(...)` using `models:/<name>@champion`.
# MAGIC - Provision a Model Serving endpoint and query it via REST through the MLflow Deployments client.
# MAGIC
# MAGIC **Databricks features showcased**
# MAGIC
# MAGIC - **Unity Catalog Model Registry** (`mlflow.set_registry_uri("databricks-uc")`) — three-part naming `<catalog>.<schema>.<model>` with UC governance and ACLs. The legacy workspace registry is deprecated; UC is now the only path.
# MAGIC - **Model aliases** (`@champion`, `@challenger`) — version-agnostic references. Downstream code reads `models:/<name>@champion`; flipping a new version into that alias is one API call, no consumer changes.
# MAGIC - **Databricks Model Serving** — provisioned via `WorkspaceClient.serving_endpoints.create(...)`, scale-to-zero by default, REST endpoint with workspace-scoped auth.
# MAGIC - **`mlflow.pyfunc.spark_udf(...)`** — batch scoring via a Spark UDF resolved from `models:/<name>@champion`. The same model URI works for batch and online.
# MAGIC - **Background-provisioning pattern** — `serving_endpoints.create(...)` returns a `Wait` handle; we do useful work and call `.result(timeout=...)` later. Same idiom used by `agents.deploy()` in Module 8.
# MAGIC
# MAGIC **Why this matters for insurtech**
# MAGIC
# MAGIC Renewal decisions happen at customer touch-time, not in nightly batch. A policyholder visits the app to update payment details — the system needs to score "is this customer at risk?" in <100ms and route them to a retention flow if so. Batch scoring (the `spark_udf` path) covers proactive nightly outreach lists; Model Serving (the REST path) covers the real-time touch-points. Both load the *same* model via `models:/<name>@champion`. Flipping the champion alias to a newly-validated version doesn't change any consumer code — the in-app risk check, the nightly batch job, and the agent in Module 8 all pick up the new version automatically.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 0, 1, 2, 3 have been run.
# MAGIC - Workspace permissions: `USE CATALOG` on `bolttech_workshop`, `CREATE MODEL` / `CREATE MODEL VERSION` on your per-user schema, ability to create serving endpoints.
# MAGIC
# MAGIC **Expected runtime**: ~7-8 minutes (endpoint cold-start dominates; other work fills the wait).
# MAGIC
# MAGIC **Compute**: Serverless or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-sdk>=0.40" \
# MAGIC   "lightgbm>=4.6"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Imports & config

# COMMAND ----------

import os
import sys

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

from config.workshop_config import (  # noqa: E402
    FULL_SCHEMA,
    CHURN_MODEL_NAME,
    CHURN_ENDPOINT,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Recover the model_ids from workshop state

# COMMAND ----------

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"

state_rows = {
    r["key"]: r["value"]
    for r in spark.table(STATE_TABLE).select("key", "value").collect()
}
baseline_model_id = state_rows["lgbm_baseline_model_id"]
tuned_model_id = state_rows["lgbm_tuned_model_id"]

print(f"Baseline LGBM model_id: {baseline_model_id}")
print(f"Tuned LGBM model_id:    {tuned_model_id}")
print(f"Will register as:       {CHURN_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2.5 Patch the logged models with the `lightgbm` dependency
# MAGIC
# MAGIC The `mlflow.sklearn.log_model()` calls in Modules 2 and 3 produced artifacts whose `requirements.txt` lists only `scikit-learn` — even though the wrapped Pipeline contains a `LGBMClassifier`. The sklearn flavor's dependency inference can't see transitive deps inside a Pipeline step. When the Model Serving container starts, `cloudpickle.load()` tries to `import lightgbm` and the container fails with:
# MAGIC
# MAGIC ```
# MAGIC ModuleNotFoundError: No module named 'lightgbm'
# MAGIC ```
# MAGIC
# MAGIC Fix: re-log each existing artifact via the sklearn flavor with `extra_pip_requirements=["lightgbm"]`, then point downstream registration at the patched `model_id`. We persist the patched IDs back to `workshop_state` so Modules 5+ pick them up.
# MAGIC
# MAGIC Long-term fix: add the same `extra_pip_requirements=` to the `log_model` calls in Modules 2 and 3 so any future end-to-end run produces correct artifacts from the start.

# COMMAND ----------

import lightgbm
import mlflow
from mlflow.models import get_model_info
from pyspark.sql import Row


def _relog_with_lightgbm(source_model_id: str, name: str) -> str:
    """Re-log an existing LoggedModel via the sklearn flavor with `lightgbm`
    explicit in the pip requirements. Returns the new model_id."""
    sk_model = mlflow.sklearn.load_model(f"models:/{source_model_id}")
    src_info = get_model_info(f"models:/{source_model_id}")

    try:
        input_example = mlflow.models.load_input_example(f"models:/{source_model_id}")
    except Exception:
        input_example = None

    with mlflow.start_run(run_name=f"{name}_relog_with_lightgbm"):
        patched = mlflow.sklearn.log_model(
            sk_model=sk_model,
            name=name,
            input_example=input_example,
            signature=src_info.signature,
            extra_pip_requirements=[f"lightgbm=={lightgbm.__version__}"],
        )
    print(f"  {name}: {source_model_id} → {patched.model_id}")
    return patched.model_id


print("Re-logging models with explicit lightgbm dependency...")
tuned_model_id = _relog_with_lightgbm(tuned_model_id, "lgbm_tuned")
baseline_model_id = _relog_with_lightgbm(baseline_model_id, "lgbm_baseline")

# Persist the patched IDs to state so Modules 5+ use the corrected artifacts.
spark.createDataFrame([
    Row(key="lgbm_tuned_model_id", value=tuned_model_id),
    Row(key="lgbm_baseline_model_id", value=baseline_model_id),
]).createOrReplaceTempView("_relog_upserts")

spark.sql(
    f"""
    MERGE INTO {STATE_TABLE} AS t
    USING _relog_upserts AS s ON t.key = s.key
    WHEN MATCHED THEN UPDATE SET value = s.value, updated_at = current_timestamp()
    """
)
print("\nState table updated with patched model_ids.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Register both models in Unity Catalog
# MAGIC
# MAGIC `mlflow.set_registry_uri("databricks-uc")` switches the registry from the legacy workspace registry to Unity Catalog. The 3-part name format is `<catalog>.<schema>.<model>`.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# Register the tuned model — this will become @champion
tuned_version = mlflow.register_model(
    model_uri=f"models:/{tuned_model_id}",
    name=CHURN_MODEL_NAME,
)
print(f"Registered tuned LGBM as {CHURN_MODEL_NAME} version {tuned_version.version}")

# Register the baseline model — this will become @challenger
baseline_version = mlflow.register_model(
    model_uri=f"models:/{baseline_model_id}",
    name=CHURN_MODEL_NAME,
)
print(f"Registered baseline LGBM as {CHURN_MODEL_NAME} version {baseline_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Set `@champion` / `@challenger` aliases
# MAGIC
# MAGIC Aliases are the MLflow 3 production-recommended way to refer to "the current production model" (vs. relying on numeric versions in downstream code).
# MAGIC
# MAGIC **Why aliases beat "production stage" labels (the MLflow 2 way).** In MLflow 2 you had a fixed set of stages (`Staging` / `Production` / `Archived`). Aliases are arbitrary string labels you choose — `@champion`, `@challenger`, `@shadow_pricing_v2`, `@sg_market`, whatever fits your promotion workflow. A model can carry multiple aliases simultaneously, version-N can be `@champion` in one workflow and `@challenger` in another, and you can A/B test by routing traffic between aliases on a single serving endpoint.
# MAGIC
# MAGIC | Pattern | Why it works |
# MAGIC | --- | --- |
# MAGIC | Batch job loads `models:/<name>@champion` | Auto-picks up the newest promoted version each run |
# MAGIC | Shadow eval loads `models:/<name>@challenger` | Compare predictions vs champion without changing consumers |
# MAGIC | Promotion = one `client.set_registered_model_alias(...)` call | No code redeploy, no env var change |
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/ml/model-registry/workflow/

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(name=CHURN_MODEL_NAME, alias="champion", version=tuned_version.version)
client.set_registered_model_alias(name=CHURN_MODEL_NAME, alias="challenger", version=baseline_version.version)

# Verify
for alias in ("champion", "challenger"):
    mv = client.get_model_version_by_alias(name=CHURN_MODEL_NAME, alias=alias)
    print(f"  @{alias:10s} → {CHURN_MODEL_NAME} version {mv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Kick off the serving endpoint (non-blocking)
# MAGIC
# MAGIC We use `serving_endpoints.create()` (returns immediately with a `Wait` handle) instead of `create_and_wait()` so the subsequent cells can do useful work while the endpoint provisions in the background.
# MAGIC
# MAGIC Idempotent: if the endpoint already exists from a prior run, we `update_config()` it instead of creating from scratch.
# MAGIC
# MAGIC **What you're getting from Databricks Model Serving** beyond "a REST URL": auto-managed container build from the LoggedModel's `requirements.txt`, workspace-scoped service-principal auth (callers from inside the workspace authenticate transparently), `scale_to_zero_enabled=True` so the endpoint costs nothing when idle, autoscaling based on QPS, integrated request logging via inference tables (Module 5 will use these), and Mosaic AI Gateway integration for rate-limiting / PII redaction / model-name routing. The DIY equivalent is your own GPU/CPU pod orchestration, a custom auth proxy, a metrics + logs sidecar, and an autoscaler — typically 4-8 weeks of platform work.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

served_entity = ServedEntityInput(
    name="churn-champion",
    entity_name=CHURN_MODEL_NAME,
    entity_version=tuned_version.version,
    workload_size="Small",
    scale_to_zero_enabled=True,
)
# `name=` is required on EndpointCoreConfigInput in databricks-sdk >=0.40; some
# older SDK versions made it optional. We pass it defensively so the config carries
# the endpoint name even though the create() call below also takes it as a kwarg.
config = EndpointCoreConfigInput(
    name=CHURN_ENDPOINT,
    served_entities=[served_entity],
)

# Check if endpoint already exists
existing = None
try:
    existing = w.serving_endpoints.get(name=CHURN_ENDPOINT)
    print(f"Endpoint {CHURN_ENDPOINT} already exists (state={existing.state.ready}); updating config.")
except Exception:
    pass

if existing is None:
    endpoint_wait = w.serving_endpoints.create(name=CHURN_ENDPOINT, config=config)
    print(f"Kicked off endpoint creation: {CHURN_ENDPOINT}")
    print("Provisioning in background; we'll wait for it after batch scoring.")
else:
    # Update config to point at the latest champion version
    endpoint_wait = w.serving_endpoints.update_config(
        name=CHURN_ENDPOINT, served_entities=[served_entity]
    )
    print(f"Updated endpoint config; new served_entities version = {tuned_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Batch score via `models:/<name>@champion` (while endpoint provisions)
# MAGIC
# MAGIC Loading by alias keeps downstream code agnostic to model version bumps — when you promote a new version to `@champion`, every consumer using `models:/.../@champion` picks it up automatically.

# COMMAND ----------

from pyspark.sql import functions as F

# Score the full customer snapshot in a single Spark job using a pyfunc UDF.
champion_uri = f"models:/{CHURN_MODEL_NAME}@champion"
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=champion_uri, env_manager="local")

features = spark.table(f"{FULL_SCHEMA}.customer_churn_features")
feature_cols = [
    c for c in features.columns if c not in ("customer_id", "snapshot_date")
]

scored = features.withColumn("predicted_churn", predict_udf(F.struct(*feature_cols)))
scored.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.batch_predictions")

display(
    spark.table(f"{FULL_SCHEMA}.batch_predictions")
    .select("customer_id", "predicted_churn")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7. Wait for the endpoint to be `READY`
# MAGIC
# MAGIC We block here until provisioning finishes. The `Wait` object's `.result(timeout=...)` polls under the hood. Typical cold-start for a small CPU model on a fresh endpoint: **~5-7 minutes**.

# COMMAND ----------

from datetime import timedelta

print(f"Waiting for {CHURN_ENDPOINT} to be READY (max 15 min)...")
endpoint_detail = endpoint_wait.result(timeout=timedelta(minutes=15))
print(f"Endpoint state: ready={endpoint_detail.state.ready}, config_update={endpoint_detail.state.config_update}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 8. Query the endpoint over REST
# MAGIC
# MAGIC Use the MLflow Deployments client — it handles auth and URL construction from the workspace context.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints

# COMMAND ----------

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")

# Pull a couple of real customer rows from the feature table to score
sample_pdf = (
    spark.table(f"{FULL_SCHEMA}.customer_churn_features")
    .select(*feature_cols)
    .limit(3)
    .toPandas()
)

payload = {"dataframe_split": {"columns": sample_pdf.columns.tolist(), "data": sample_pdf.values.tolist()}}
result = deploy_client.predict(endpoint=CHURN_ENDPOINT, inputs=payload)
print("Endpoint response:")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 9. Persist endpoint name for downstream modules

# COMMAND ----------

from pyspark.sql import Row

spark.createDataFrame([
    Row(key="churn_endpoint_name", value=CHURN_ENDPOINT),
    Row(key="churn_model_uc_name", value=CHURN_MODEL_NAME),
    Row(key="churn_champion_version", value=str(tuned_version.version)),
]).createOrReplaceTempView("_state_upserts")

spark.sql(
    f"""
    MERGE INTO {STATE_TABLE} AS t
    USING _state_upserts AS s ON t.key = s.key
    WHEN MATCHED THEN UPDATE SET value = s.value, updated_at = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (key, value, updated_at) VALUES (s.key, s.value, current_timestamp())
    """
)
display(spark.table(STATE_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just learned**
# MAGIC
# MAGIC - Registering an MLflow 3 `LoggedModel` in Unity Catalog: `mlflow.set_registry_uri("databricks-uc")` + `mlflow.register_model(model_uri="models:/<model_id>", name="<cat>.<sch>.<model>")`.
# MAGIC - Aliases as the version-agnostic reference: `@champion` / `@challenger`. Loading via `models:/<name>@<alias>` makes downstream code immune to version churn.
# MAGIC - Background-provisioning pattern: fire `serving_endpoints.create()`, do other work, `.result(timeout=...)` at the end.
# MAGIC - Two ways to score: `mlflow.pyfunc.spark_udf` for batch (no endpoint needed) vs `mlflow.deployments.get_deploy_client("databricks").predict(...)` for REST.
# MAGIC
# MAGIC **What you'd build without Databricks**
# MAGIC
# MAGIC Stand up your own model registry (MLflow OSS server + database + artifact store), build a custom alias / stage layer with versioned routing, containerize each model version yourself with Docker + the right base image + pip requirements, run a Kubernetes inference service with autoscaling rules, build an auth proxy that resolves workspace identities to model permissions, and add a separate request-logging pipeline so you can later detect drift. Three to four full quarters of platform engineering replaced by `register_model(...)` + `serving_endpoints.create(...)`.
# MAGIC
# MAGIC **How this composes in production**
# MAGIC
# MAGIC The REST endpoint you just provisioned is the production scoring path — Module 8's retention agent calls it as a tool via `DatabricksServingEndpoint(endpoint_name=CHURN_ENDPOINT)` with auto-auth, and Module 10's capstone uses the same endpoint for end-of-pipeline scoring. The inference table that Model Serving writes (when enabled) is what Module 5's monitoring would attach to in a real deployment. The `@champion` alias is the contract: when a future retraining run produces a better tuned model, you re-register it, flip the alias, and every consumer — batch UDF, REST endpoint, agent tool — picks up the new version with zero code change.
# MAGIC
# MAGIC **What's next — Module 5: Lakehouse Monitoring**
# MAGIC
# MAGIC Module 5 simulates an inference table with synthetic drift and sets up a Lakehouse monitor against the model you just deployed. Open `modules/05_monitoring/05_monitoring.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [Manage model lifecycle in UC](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/)
# MAGIC - [Create custom serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints)
# MAGIC - [Score custom model endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints)
# MAGIC - [Inference tables for served models](https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables)
