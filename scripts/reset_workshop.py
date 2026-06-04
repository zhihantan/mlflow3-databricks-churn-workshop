# Databricks notebook source
# MAGIC %md
# MAGIC # Reset Workshop — tear down all participant artifacts for a clean re-run
# MAGIC
# MAGIC Drops, in dependency-safe order:
# MAGIC
# MAGIC 1. **Agent serving endpoint** (Module 8) — may take a minute
# MAGIC 2. **Churn serving endpoint** (Module 4)
# MAGIC 3. **Vector Search index** (Module 7)
# MAGIC 4. **Vector Search endpoint** (Module 6/7)
# MAGIC 5. **Registered models in UC** (`bolttech_retention_agent`, `bolttech_churn_model`)
# MAGIC 6. **Prompt Registry entries** (Module 6, 7)
# MAGIC 7. **Lakehouse monitor** on the inference table (Module 5)
# MAGIC 8. **UC schema** (cascade — drops all tables/views inside)
# MAGIC
# MAGIC **Idempotent** — skips resources that don't exist. Re-running is safe.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-sdk>=0.40" \
# MAGIC   "databricks-vectorsearch>=0.50" \
# MAGIC   "databricks-lakehouse-monitoring"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(_nb_path))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

from config.workshop_config import (  # noqa: E402
    FULL_SCHEMA,
    CHURN_MODEL_NAME,
    AGENT_MODEL_NAME,
    CHURN_ENDPOINT,
    AGENT_ENDPOINT,
    VS_ENDPOINT,
    VS_INDEX,
    INFERENCE_TABLE,
    SUMMARY_PROMPT_NAME,
    RAG_PROMPT_NAME,
    EMAIL_PROMPT_NAME,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Serving endpoints
# MAGIC
# MAGIC The agent endpoint (from `agents.deploy()`) has an auto-generated name we read from `workshop_state` if present; otherwise we attempt `AGENT_ENDPOINT` from config. Same for the churn endpoint.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"
try:
    state_rows = {r["key"]: r["value"] for r in spark.table(STATE_TABLE).select("key", "value").collect()}
except Exception:
    state_rows = {}

endpoints_to_drop = [
    state_rows.get("agent_endpoint_name", AGENT_ENDPOINT),
    state_rows.get("churn_endpoint_name", CHURN_ENDPOINT),
]

for name in endpoints_to_drop:
    if not name:
        continue
    try:
        w.serving_endpoints.delete(name=name)
        print(f"  Deleted serving endpoint: {name}")
    except Exception as exc:
        print(f"  Skipped serving endpoint {name}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Vector Search index + endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

try:
    vsc.delete_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
    print(f"  Deleted VS index: {VS_INDEX}")
except Exception as exc:
    print(f"  Skipped VS index: {exc}")

try:
    vsc.delete_endpoint(name=VS_ENDPOINT)
    print(f"  Deleted VS endpoint: {VS_ENDPOINT}")
except Exception as exc:
    print(f"  Skipped VS endpoint: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Registered models in UC

# COMMAND ----------

from mlflow import MlflowClient

mlflow_client = MlflowClient(registry_uri="databricks-uc")

for model_name in (AGENT_MODEL_NAME, CHURN_MODEL_NAME):
    try:
        mlflow_client.delete_registered_model(name=model_name)
        print(f"  Deleted registered model: {model_name}")
    except Exception as exc:
        print(f"  Skipped registered model {model_name}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Prompt Registry entries

# COMMAND ----------

import mlflow.genai

for prompt_name in (SUMMARY_PROMPT_NAME, RAG_PROMPT_NAME, EMAIL_PROMPT_NAME):
    try:
        mlflow.genai.delete_prompt(name=prompt_name)
        print(f"  Deleted prompt: {prompt_name}")
    except Exception as exc:
        print(f"  Skipped prompt {prompt_name}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4b — Scheduled scorers (Module 9 production monitoring)
# MAGIC
# MAGIC Stops + deletes any scheduled scorers registered on the workshop experiment by Module 9 §9
# MAGIC (`prod_safety`, `prod_bolttech_voice`, or any others). Idempotent — no-op if none exist.

# COMMAND ----------

# Ref: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/production-quality-monitoring
try:
    import mlflow
    from config.workshop_config import EXPERIMENT_PATH  # scheduled scorers are experiment-scoped

    mlflow.set_experiment(EXPERIMENT_PATH)
    from mlflow.genai.scorers import list_scorers, delete_scorer

    for _sc in list_scorers():
        try:
            if hasattr(_sc, "stop"):
                _sc.stop()
        except Exception as exc:
            print(f"  stop skipped for {_sc.name}: {exc}")
        try:
            delete_scorer(name=_sc.name)
            print(f"  Deleted scheduled scorer: {_sc.name}")
        except Exception as exc:
            print(f"  Skipped scorer {_sc.name}: {exc}")
except Exception as exc:
    print(f"  Scorer cleanup skipped (API unavailable or none registered): {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Lakehouse monitor

# COMMAND ----------

try:
    import databricks.lakehouse_monitoring as lm  # type: ignore

    try:
        lm.delete_monitor(table_name=INFERENCE_TABLE)
        print(f"  Deleted monitor on: {INFERENCE_TABLE}")
    except Exception as exc:
        print(f"  Skipped monitor: {exc}")
except ImportError:
    print("  databricks.lakehouse_monitoring not available; skipping monitor cleanup")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — UC schema cascade
# MAGIC
# MAGIC `DROP SCHEMA ... CASCADE` removes all tables/views/feature-tables inside the per-user schema in one shot.

# COMMAND ----------

try:
    spark.sql(f"DROP SCHEMA IF EXISTS {FULL_SCHEMA} CASCADE")
    print(f"  Dropped schema: {FULL_SCHEMA}")
except Exception as exc:
    print(f"  Schema drop failed: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC The workshop is fully reset. To re-run:
# MAGIC
# MAGIC 1. Open `setup/00_setup_and_synthetic_data.py`
# MAGIC 2. Run all cells
# MAGIC 3. Continue through Modules 1 → 10
