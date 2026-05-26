# Databricks notebook source
# MAGIC %md
# MAGIC # Module 10 — Capstone: Batch Score + Agent Drafted Outreach
# MAGIC
# MAGIC The full workshop in one notebook. We:
# MAGIC
# MAGIC 1. Batch-score the customer snapshot via the Module 4 churn model.
# MAGIC 2. Rank top-10 highest-risk customers.
# MAGIC 3. Invoke the deployed Module 8 retention agent on each one to produce a personalized retention email.
# MAGIC 4. Display the drafted outreach in a styled table.
# MAGIC 5. Close on a productionization discussion.
# MAGIC
# MAGIC No new APIs — everything you've learned, stitched together.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 4 and 8 have been run, and **both their serving endpoints are READY**.
# MAGIC
# MAGIC **Expected runtime**: ~3-5 minutes (10 agent calls × ~5-10s each).
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-sdk>=0.40" \
# MAGIC   "openai>=1.50" \
# MAGIC   "lightgbm>=4.6" \
# MAGIC   "scikit-learn>=1.6"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports & recover workshop state

# COMMAND ----------

import os
import sys
import time

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

from config.workshop_config import (  # noqa: E402
    FULL_SCHEMA,
    CHURN_MODEL_NAME,
    EXPERIMENT_PATH,
    print_config,
)

print_config()

# Set the MLflow experiment + enable OpenAI autotracing so every agent invocation
# below produces a captured trace (visible in the experiment's Traces tab).
import mlflow

mlflow.set_experiment(EXPERIMENT_PATH)
try:
    # `mlflow.openai.autolog()` requires the `openai` package to be importable for
    # the patch hook to register. If install raced or the package isn't available
    # for any reason, skip autolog gracefully — the @mlflow.trace decorator on
    # call_agent still produces traces; only the auto-nested OpenAI sub-spans are
    # missed.
    mlflow.openai.autolog()
except Exception as exc:
    print(f"(mlflow.openai.autolog skipped: {type(exc).__name__}: {exc}) — manual @mlflow.trace traces still work")

# COMMAND ----------

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"
state_rows = {r["key"]: r["value"] for r in spark.table(STATE_TABLE).select("key", "value").collect()}
churn_endpoint = state_rows["churn_endpoint_name"]
agent_endpoint = state_rows.get("agent_endpoint_name", "")
agent_model_id = state_rows.get("agent_model_id", "")
agent_review_app_url = state_rows.get("agent_review_app_url", "")

print(f"Churn endpoint: {churn_endpoint}")
print(f"Agent endpoint: {agent_endpoint}")
print(f"Agent model_id (local fallback): {agent_model_id}")
print(f"Review App URL: {agent_review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Batch-score the customer snapshot via the champion model
# MAGIC
# MAGIC Same `models:/.../@champion` Spark UDF as Module 4 — we already have the predictions table, just refresh it to make sure we're scoring against the latest features.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

champion_uri = f"models:/{CHURN_MODEL_NAME}@champion"
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=champion_uri, env_manager="local")

features = spark.table(f"{FULL_SCHEMA}.customer_churn_features")
feature_cols = [c for c in features.columns if c not in ("customer_id", "snapshot_date")]

scored = (
    features
    .withColumn("predicted_churn", predict_udf(F.struct(*feature_cols)))
)
display(
    scored
    .select("customer_id", "country", "plan_tier", "predicted_churn", "payment_failures_60d", "pending_claims_90d", "support_ticket_count_30d")
    .orderBy(F.col("predicted_churn").desc())
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pick the top-10 highest-risk customers

# COMMAND ----------

top_10 = (
    scored
    .orderBy(F.col("predicted_churn").desc())
    .limit(10)
    .select("customer_id", "country", "plan_tier", "predicted_churn", "payment_failures_60d", "pending_claims_90d")
    .toPandas()
)
display(top_10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Invoke the retention agent on each
# MAGIC
# MAGIC Prefer the deployed endpoint (the production path); fall back to the local-loaded agent if the endpoint isn't yet `READY` (Module 8 may have timed out waiting for cold-start).

# COMMAND ----------

import mlflow.deployments
from databricks.sdk import WorkspaceClient

deploy_client = mlflow.deployments.get_deploy_client("databricks")
w = WorkspaceClient()


def _endpoint_ready(name: str) -> bool:
    try:
        ep = w.serving_endpoints.get(name=name)
        return str(getattr(ep.state, "ready", "")).upper().endswith("READY")
    except Exception:
        return False


use_deployed = bool(agent_endpoint) and _endpoint_ready(agent_endpoint)
print(f"Agent endpoint ready: {use_deployed}")
if not use_deployed:
    # Load local agent as fallback
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    os.environ.setdefault("DATABRICKS_HOST", ctx.apiUrl().get())
    os.environ.setdefault("DATABRICKS_TOKEN", ctx.apiToken().get())
    from config.workshop_config import CHURN_ENDPOINT as _CE, VS_ENDPOINT as _VE, VS_INDEX as _VI, CHAT_MODEL as _CM
    os.environ["CHURN_ENDPOINT"] = _CE
    os.environ["VS_ENDPOINT"] = _VE
    os.environ["VS_INDEX"] = _VI
    os.environ["CHAT_MODEL"] = _CM
    local_agent = mlflow.pyfunc.load_model(f"models:/{agent_model_id}")
    print(f"Falling back to local-loaded agent model_id={agent_model_id}")


from mlflow.entities import SpanType


@mlflow.trace(name="capstone_call_agent", span_type=SpanType.CHAIN)
def call_agent(customer_id: str) -> str:
    """One traced span per customer — visible in the experiment Traces tab.

    The inner agent/openai calls auto-nest underneath when running via the local
    agent path (mlflow.openai.autolog catches them). For the deployed-endpoint
    path, the agent endpoint emits its own trace into the experiment as well.
    """
    payload = {"input": [{"role": "user", "content": f"Draft a retention email for customer {customer_id}"}]}
    if use_deployed:
        resp = deploy_client.predict(endpoint=agent_endpoint, inputs=payload)
    else:
        resp = local_agent.predict(payload)
    parts: list[str] = []
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    parts.append(c.get("text", ""))
    return "\n".join(parts).strip()


# COMMAND ----------

# MAGIC %md
# MAGIC Sequential invocation. 10 customers × ~5-10s each. For real production scale you'd parallelize this in a Job task or use `mlflow.deployments` batch inference primitives.

# COMMAND ----------

results: list[dict] = []
for _, row in top_10.iterrows():
    t0 = time.time()
    email = call_agent(row["customer_id"])
    elapsed = time.time() - t0
    results.append({
        "customer_id": row["customer_id"],
        "country": row["country"],
        "plan_tier": row["plan_tier"],
        "predicted_churn": float(row["predicted_churn"]),
        "payment_failures_60d": int(row["payment_failures_60d"]),
        "pending_claims_90d": int(row["pending_claims_90d"]),
        "agent_latency_s": round(elapsed, 2),
        "drafted_email": email,
    })
    print(f"  {row['customer_id']} ({elapsed:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Display drafted outreach

# COMMAND ----------

import pandas as pd

display_pdf = pd.DataFrame(results)
display(display_pdf.drop(columns=["drafted_email"]))

# COMMAND ----------

# Render the actual emails one-by-one for readability
for r in results:
    print(f"\n=== {r['customer_id']} ({r['country']}, {r['plan_tier']}, risk={r['predicted_churn']}) ===")
    print(r["drafted_email"])
    print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist the capstone results

# COMMAND ----------

(
    spark.createDataFrame(display_pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{FULL_SCHEMA}.capstone_retention_emails")
)
print(f"Saved to {FULL_SCHEMA}.capstone_retention_emails")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Productionization discussion
# MAGIC
# MAGIC What you'd add to ship this for real:
# MAGIC
# MAGIC | Concern | How to address |
# MAGIC | --- | --- |
# MAGIC | **Scheduling** | Move this notebook into a Databricks **Job** with a cron schedule (daily at 09:00 local). Use Job tasks to chain Module 0 → 1 → 2 → ... if you want full retraining cadence. |
# MAGIC | **Human-in-the-loop review** | Module 8's `agents.deploy()` provisioned a Review App — point your CS team at the URL above. They approve/edit drafts before sending. |
# MAGIC | **Drift monitoring** | Module 5's Lakehouse Monitor is already running. Add alerts (Databricks Alerts → Slack / PagerDuty) when KS / JS distance on `payment_failures_60d` exceeds a threshold. |
# MAGIC | **Cost control** | FMAPI is pay-per-token. The agent at full capstone scale (~10 outreach emails / day) is < $1 / day. Watch out when scaling to thousands of customers per run — use `scale_to_zero_enabled=True` (already on) and budget alerts on the workspace. |
# MAGIC | **Governance** | Put the agent endpoint behind **Unity AI Gateway** (Beta) to centralize logging, permissions, and guardrails. |
# MAGIC | **A/B testing** | Use the `@champion` / `@challenger` aliasing pattern to A/B test new model versions. Route traffic via `served_entities` weights on the serving endpoint. |
# MAGIC | **Feedback loop** | The deployed agent endpoint's auto-captured inference table can be re-used as a training signal source (e.g., did the email actually retain the customer?). Close the loop by joining outcomes back into the next training run's labels. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap — the full workshop
# MAGIC
# MAGIC In 10 modules you built:
# MAGIC
# MAGIC 1. **Module 0** — Synthetic bolttech insurtech data in Unity Catalog.
# MAGIC 2. **Module 1** — UC feature table with point-in-time correctness.
# MAGIC 3. **Module 2** — Baseline LR + LightGBM with the new MLflow 3 `LoggedModel` entity.
# MAGIC 4. **Module 3** — Optuna tuning + `mlflow.evaluate` with a custom business metric.
# MAGIC 5. **Module 4** — UC Model Registry + Model Serving endpoint (background-provisioning pattern).
# MAGIC 6. **Module 5** — Lakehouse Monitoring on an inference table with synthetic drift.
# MAGIC 7. **Module 6** — MLflow Tracing + Prompt Registry foundations.
# MAGIC 8. **Module 7** — RAG over support tickets via Vector Search.
# MAGIC 9. **Module 8** — `ResponsesAgent` with two tools, logged + UC-registered + actually deployed.
# MAGIC 10. **Module 9** — `mlflow.genai.evaluate` with built-in + custom scorers, iteration loop demo.
# MAGIC 11. **Module 10** — End-to-end batch scoring + agent-drafted retention outreach.
# MAGIC
# MAGIC The thread holding it all together: MLflow 3's `LoggedModel`, Prompt Registry, Tracing, and the new `mlflow.genai.evaluate` — plus Databricks Unity Catalog as the durable home for data, features, models, prompts, and inference traces.
# MAGIC
# MAGIC When you're done playing, `scripts/reset_workshop.py` tears it all down for a clean re-run.
# MAGIC
# MAGIC **Thanks for working through this — happy MLflow-ing.**
