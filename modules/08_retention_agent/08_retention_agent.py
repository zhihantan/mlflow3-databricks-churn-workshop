# Databricks notebook source
# MAGIC %md
# MAGIC # Module 08 — Retention Outreach Agent (`ResponsesAgent` + deploy)
# MAGIC
# MAGIC The most ambitious module in the workshop. We author a tool-using agent, log it, register it in Unity Catalog, and **actually deploy it** via `agents.deploy()` — getting a real Model Serving endpoint + Review App URL + inference table + tracing for free. The 8-12 min agent-endpoint cold-start is absorbed in-cell by exercising the local-loaded copy of the agent first.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Author a `mlflow.pyfunc.ResponsesAgent` subclass that uses the OpenAI Agents SDK as its inner loop.
# MAGIC - Define two tools that call back into earlier workshop artifacts (Module 4 endpoint + Module 7 VS index).
# MAGIC - Log the agent via Models-from-Code with `resources=[...]` declarations for auto-auth.
# MAGIC - Register it in UC and call `agents.deploy()` end-to-end.
# MAGIC - Test locally first, then query the deployed endpoint.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 0, 4, 6, 7 have all been run. (Module 4's churn endpoint and Module 7's VS index must exist.)
# MAGIC
# MAGIC **Expected runtime**: ~9 minutes (background-provisioning pattern absorbs the agent endpoint cold-start).
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "openai>=1.50" \
# MAGIC   "openai-agents" \
# MAGIC   "databricks-agents" \
# MAGIC   "databricks-vectorsearch>=0.50" \
# MAGIC   "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports & config

# COMMAND ----------

import json
import os
import sys
import tempfile
import time

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir = os.path.dirname(_nb_path)
# modules/<module>/<notebook> → 3 dirnames to repo root
_repo_root_rel = os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

_agent_module_path = (_nb_dir if _nb_dir.startswith("/Workspace") else "/Workspace" + _nb_dir) + "/agent.py"

from config.workshop_config import (  # noqa: E402
    FULL_SCHEMA,
    CHURN_ENDPOINT,
    CHURN_MODEL_NAME,
    VS_ENDPOINT,
    VS_INDEX,
    CHAT_MODEL,
    AGENT_MODEL_NAME,
    AGENT_ENDPOINT,
    EXPERIMENT_PATH,
    print_config,
)

# Set Databricks workspace credentials in env vars NOW, BEFORE we call log_model. The
# log_model signature-inference path triggers a predict() against the input_example,
# which in turn needs OPENAI_API_KEY (the OpenAI Agents SDK talks to FMAPI through
# the standard OpenAI client). agent.py's _configure_openai_for_databricks() reads
# DATABRICKS_HOST/_TOKEN and exports OPENAI_BASE_URL/_API_KEY — but only if those
# DATABRICKS_* env vars are populated in this process first.
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"] = _ctx.apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().get()
# Also export the OpenAI client env vars directly so the deployed endpoint inherits
# them (and so any module-level openai client instantiation picks them up).
os.environ["OPENAI_BASE_URL"] = f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints"
os.environ["OPENAI_API_KEY"] = os.environ["DATABRICKS_TOKEN"]

print_config()
print(f"\nAgent module: {_agent_module_path}")
print(f"Exists: {os.path.exists(_agent_module_path)}")
print(f"OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the customer-features lookup artifact
# MAGIC
# MAGIC The agent's `churn_score_tool` needs the features to POST to the churn endpoint. We can't run Spark inside a Model Serving endpoint, so we pre-bake a small JSON artifact mapping `customer_id → features_dict` and bundle it with the logged model. For the workshop we include only the top-200 highest-risk customers (Module 4's batch_predictions sorted by predicted_churn) to keep the artifact small (~50KB).

# COMMAND ----------

from pyspark.sql import functions as F

batch_preds = spark.table(f"{FULL_SCHEMA}.batch_predictions")
features_table = spark.table(f"{FULL_SCHEMA}.customer_churn_features")

# Join + take top-200 highest-risk
top_risk = (
    batch_preds.select("customer_id", "predicted_churn")
    .orderBy(F.col("predicted_churn").desc())
    .limit(200)
)
joined = top_risk.join(features_table, on="customer_id", how="left")

feature_cols = [c for c in features_table.columns if c not in ("customer_id", "snapshot_date")]
features_pdf = joined.select("customer_id", *feature_cols).toPandas()
features_lookup = {
    row["customer_id"]: {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if k != "customer_id"}
    for _, row in features_pdf.iterrows()
}

# Write to a temp file we can pass as an artifact to log_model
features_json_path = tempfile.mktemp(suffix="_customer_features.json")
with open(features_json_path, "w") as fh:
    json.dump(features_lookup, fh)

print(f"Features lookup: {len(features_lookup):,} customers → {os.path.getsize(features_json_path):,} bytes")
print(f"Sample customer: {list(features_lookup.keys())[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log the agent (Models-from-Code) — fast
# MAGIC
# MAGIC We pass `python_model=<path to agent.py>` so MLflow ingests the source file and uses `mlflow.models.set_model(...)` (at the bottom of `agent.py`) to identify the entry-point object.
# MAGIC
# MAGIC The `resources=[...]` list is critical — it declares which Databricks resources the agent needs at inference time, so when this model is deployed via `agents.deploy()`, Databricks Model Serving injects credentials so the agent can reach those resources without manual token plumbing.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.openai.autolog()

# Serialize endpoint/index identifiers as a JSON artifact so the agent's load_context
# picks them up regardless of environment. This is more robust than env vars across
# the local-vs-deployed boundary.
agent_config = {
    "CHURN_ENDPOINT": CHURN_ENDPOINT,
    "VS_ENDPOINT": VS_ENDPOINT,
    "VS_INDEX": VS_INDEX,
    "CHAT_MODEL": CHAT_MODEL,
}
agent_config_path = tempfile.mktemp(suffix="_agent_config.json")
with open(agent_config_path, "w") as fh:
    json.dump(agent_config, fh)

input_example = {
    "input": [
        {"role": "user", "content": "Draft a retention email for customer CUST_000001"},
    ]
}

with mlflow.start_run(run_name="retention_agent_log") as run:
    logged = mlflow.pyfunc.log_model(
        python_model=_agent_module_path,
        name="retention_agent",
        artifacts={
            "customer_features": features_json_path,
            "agent_config": agent_config_path,
        },
        input_example=input_example,
        resources=[
            DatabricksServingEndpoint(endpoint_name=CHAT_MODEL),
            DatabricksServingEndpoint(endpoint_name=CHURN_ENDPOINT),
            DatabricksVectorSearchIndex(index_name=VS_INDEX),
        ],
        pip_requirements=[
            "mlflow[databricks]>=3.12,<4",
            "openai>=1.50",
            "openai-agents",
            "databricks-vectorsearch>=0.50",
            "databricks-sdk>=0.40",
        ],
    )

print(f"Logged agent model_id: {logged.model_id}")
print(f"Logged model_uri:     {logged.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register in Unity Catalog + kick off `agents.deploy()`
# MAGIC
# MAGIC `agents.deploy()` returns immediately with a deployment handle; the underlying Model Serving endpoint provisions in the background (~8-12 min for an agent endpoint with these dependencies). We grab the deployment handle, then move on to local testing while the cluster spins up.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent

# COMMAND ----------

from databricks import agents

mlflow.set_registry_uri("databricks-uc")
uc_info = mlflow.register_model(model_uri=logged.model_uri, name=AGENT_MODEL_NAME)
print(f"Registered in UC: {AGENT_MODEL_NAME} version {uc_info.version}")

# Kick off the deployment. This call returns fast; the endpoint provisions in background.
# The agent gets its endpoint/index identifiers from the `agent_config` artifact
# baked into the logged model, so no environment_vars are needed at deploy time.
try:
    deployment = agents.deploy(
        model_name=AGENT_MODEL_NAME,
        model_version=uc_info.version,
        scale_to_zero_enabled=True,
    )
except TypeError as exc:
    # Older SDK versions may not support all kwargs — fall back to minimal call
    print(f"agents.deploy() kwargs incompatibility ({exc}); retrying with minimal call.")
    deployment = agents.deploy(AGENT_MODEL_NAME, uc_info.version)

print(f"\nDeployment kicked off:")
print(f"  endpoint_url:    {getattr(deployment, 'endpoint_url', '<unknown>')}")
print(f"  review_app_url:  {getattr(deployment, 'review_app_url', '<unknown>')}")
print(f"  endpoint_name:   {getattr(deployment, 'endpoint_name', '<unknown>')}")
print(f"\nProvisioning in background — we'll exercise the local copy while it warms up.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Local test — load + invoke the agent in-process
# MAGIC
# MAGIC Fast iteration loop: load the logged agent via `mlflow.pyfunc.load_model(...)`, hit it with a sample request, see traces appear in the experiment. No endpoint, no deploy wait. This is the workflow you'd use during day-to-day development.

# COMMAND ----------

# Env vars were already set in cell 2 (before log_model) — no setdefault needed here.
local_agent = mlflow.pyfunc.load_model(f"models:/{logged.model_id}")

# Pick a known-high-risk customer ID from our features lookup so the tools have data to find
sample_customer_id = list(features_lookup.keys())[0]
print(f"Testing on customer: {sample_customer_id}\n")

local_response = local_agent.predict(
    {"input": [{"role": "user", "content": f"Draft a retention email for customer {sample_customer_id}"}]}
)
print("=== Local agent response ===")
for item in local_response.get("output", []):
    if item.get("type") == "message":
        for part in item.get("content", []):
            if part.get("type") in ("output_text", "text"):
                print(part.get("text", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run the agent on two more sample customers
# MAGIC
# MAGIC Confirms the tools generalize and shows the variety of outputs the model produces.

# COMMAND ----------

for cid in list(features_lookup.keys())[1:3]:
    print(f"\n=== Customer {cid} ===")
    resp = local_agent.predict(
        {"input": [{"role": "user", "content": f"Draft a retention email for customer {cid}"}]}
    )
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    print(part.get("text", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Wait for the deployed endpoint to be `READY`
# MAGIC
# MAGIC By now the agent endpoint has been provisioning for several minutes while we exercised the local copy. Poll until ready (or timeout — in which case Module 10 will pick up the work when the endpoint finishes).

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
target_endpoint = deployment.endpoint_name

deadline = time.time() + 900  # 15 min budget
while True:
    ep = w.serving_endpoints.get(name=target_endpoint)
    state = ep.state
    ready_flag = getattr(state, "ready", "UNKNOWN")
    config_update = getattr(state, "config_update", "UNKNOWN")
    if str(ready_flag) == "READY" or str(ready_flag).upper().endswith("READY"):
        print(f"  endpoint READY (config_update={config_update})")
        break
    if time.time() > deadline:
        print(f"  endpoint not READY within 15 min — leaving Module 10 to pick up.")
        print(f"  current state: ready={ready_flag}, config_update={config_update}")
        break
    print(f"  ready={ready_flag}, config_update={config_update}, sleeping 20s...")
    time.sleep(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Query the deployed endpoint
# MAGIC
# MAGIC If the endpoint reached `READY`, send the same prompt we used locally to compare responses. If it timed out, this cell will surface the URL so participants can return to it once provisioning finishes.

# COMMAND ----------

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")

try:
    live_response = deploy_client.predict(
        endpoint=target_endpoint,
        inputs={"input": [{"role": "user", "content": f"Draft a retention email for customer {sample_customer_id}"}]},
    )
    print("=== Deployed agent response ===")
    for item in live_response.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    print(part.get("text", ""))
except Exception as exc:
    print(f"Endpoint not yet queryable: {exc}")
    print(f"Endpoint URL (revisit later): {deployment.endpoint_url}")
    print(f"Review App URL:               {deployment.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Persist agent endpoint info for Modules 9 & 10

# COMMAND ----------

from pyspark.sql import Row

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"
spark.createDataFrame([
    Row(key="agent_model_id", value=logged.model_id),
    Row(key="agent_uc_name", value=AGENT_MODEL_NAME),
    Row(key="agent_uc_version", value=str(uc_info.version)),
    Row(key="agent_endpoint_name", value=target_endpoint),
    Row(key="agent_endpoint_url", value=deployment.endpoint_url or ""),
    Row(key="agent_review_app_url", value=deployment.review_app_url or ""),
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
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just built**
# MAGIC
# MAGIC - A `mlflow.pyfunc.ResponsesAgent` subclass (`agent.py`) with two tools wired to the Module 4 churn endpoint and the Module 7 VS index.
# MAGIC - Models-from-Code logging with `resources=[...]` so the deployed endpoint has auto-auth into the downstream Databricks resources.
# MAGIC - A UC-registered agent at `<schema>.bolttech_retention_agent` and a real `agents.deploy()` provisioned endpoint with a Review App.
# MAGIC
# MAGIC **What's next — Module 9: GenAI Evaluation**
# MAGIC
# MAGIC Module 9 runs `mlflow.genai.evaluate` on the **locally-loaded** agent (faster, no dependency on deploy timing) with built-in scorers + a custom `Guidelines` scorer for bolttech voice. Open `modules/09_genai_evaluation/09_genai_evaluation.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [Author a `ResponsesAgent`](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent)
# MAGIC - [Log an agent](https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent)
# MAGIC - [Deploy an agent](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent)
# MAGIC - [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
