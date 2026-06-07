# Databricks notebook source
# MAGIC %md
# MAGIC # Module 11 — Teardown (destroy all workshop resources)
# MAGIC ### Run separately, on demand — NOT part of the e2e job
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > **TL;DR** — Idempotent, dependency-safe teardown of *everything* this workshop creates for the current user: serving endpoints, Vector Search, UC registered models, Prompt Registry entries, scheduled scorers, the Lakehouse monitor, the **MLflow experiment** (so its path is freed for a clean UC-trace re-bind), and finally the per-user UC schema (cascade). A confirmation widget guards against accidental runs.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Why this is a standalone module, not an e2e-job task.** Destroying resources is intentionally a manual, opt-in action — wiring it into the chained `MLFlow Workshop e2e job` would nuke the workshop the moment the job finished. It is excluded from both job definitions (`resources/workshop_e2e_job.yml` and `scripts/deploy_workshop_job.py`) on purpose. Open and `Run All` this notebook only when you want a clean slate.
# MAGIC
# MAGIC **Deletes the MLflow experiment too.** A UC trace destination can only bind to a *trace-free* experiment, so leaving the old experiment in place is exactly what blocks UC trace storage on a re-run. Deleting it here frees the path so Module 0 can re-bind cleanly. This is the single, canonical teardown for the workshop.
# MAGIC
# MAGIC **What it removes (in dependency-safe order)**
# MAGIC
# MAGIC 1. **Serving endpoints** — agent (Module 8) + churn (Module 4)
# MAGIC 2. **Vector Search** — index then endpoint (Modules 6/7)
# MAGIC 3. **Registered models in UC** — `bolttech_retention_agent`, `bolttech_churn_model`
# MAGIC 4. **Prompt Registry entries** — summary / RAG / email prompts (Modules 6, 7)
# MAGIC 5. **Scheduled scorers** — stop + delete production monitors (Module 9 §9)
# MAGIC 6. **Lakehouse monitor** — on the inference table (Module 5)
# MAGIC 7. **MLflow experiment** — deletes it so the path is free for a fresh UC-bound re-run
# MAGIC 8. **UC schema (cascade)** — drops every table/view/feature-table, incl. the `mlflow_traces_*` UC trace tables and `workshop_state`
# MAGIC
# MAGIC **Idempotent** — every step skips resources that don't exist, so re-running is safe.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-sdk>=0.40" \
# MAGIC   "databricks-vectorsearch>=0.50" \
# MAGIC   "databricks-lakehouse-monitoring"
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
    AGENT_MODEL_NAME,
    CHURN_ENDPOINT,
    AGENT_ENDPOINT,
    VS_ENDPOINT,
    VS_INDEX,
    INFERENCE_TABLE,
    EXPERIMENT_PATH,
    SUMMARY_PROMPT_NAME,
    RAG_PROMPT_NAME,
    EMAIL_PROMPT_NAME,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Safety confirmation
# MAGIC
# MAGIC This module **permanently destroys** the resources above for the current user. To prevent an
# MAGIC accidental `Run All`, the destructive steps only execute when you explicitly confirm the target
# MAGIC schema. Set the **`confirm_schema`** widget at the top of the notebook to the exact `FULL_SCHEMA`
# MAGIC value printed above (e.g. `bolttech_workshop.churn_<you>`), then run the rest of the notebook.

# COMMAND ----------

dbutils.widgets.text("confirm_schema", "", "Type the FULL_SCHEMA to confirm teardown")
_confirm = dbutils.widgets.get("confirm_schema").strip()

if _confirm != FULL_SCHEMA:
    dbutils.notebook.exit(
        f"Teardown NOT confirmed. Set the 'confirm_schema' widget to '{FULL_SCHEMA}' "
        f"(got '{_confirm or '<empty>'}') and re-run. No resources were touched."
    )

print(f"Confirmed — tearing down all workshop resources for: {FULL_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Serving endpoints
# MAGIC
# MAGIC The agent endpoint (from `agents.deploy()`) may have an auto-generated name written to
# MAGIC `workshop_state`; we prefer that and fall back to the config name. Same for the churn endpoint.

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
# MAGIC ---
# MAGIC ## 4. Vector Search index + endpoint

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
# MAGIC ---
# MAGIC ## 5. Registered models in UC

# COMMAND ----------

from mlflow import MlflowClient

uc_client = MlflowClient(registry_uri="databricks-uc")

for model_name in (AGENT_MODEL_NAME, CHURN_MODEL_NAME):
    try:
        uc_client.delete_registered_model(name=model_name)
        print(f"  Deleted registered model: {model_name}")
    except Exception as exc:
        print(f"  Skipped registered model {model_name}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Prompt Registry entries

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
# MAGIC ---
# MAGIC ## 7. Scheduled scorers (Module 9 production monitoring)
# MAGIC
# MAGIC Stop + delete any scheduled scorers registered on the workshop experiment (`prod_safety`,
# MAGIC `prod_bolttech_voice`, …). Runs **before** the experiment is deleted, since scorers are
# MAGIC experiment-scoped. Idempotent — no-op if none exist.

# COMMAND ----------

# Ref: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/production-quality-monitoring
try:
    import mlflow

    mlflow.set_experiment(EXPERIMENT_PATH)  # scope scorer lookups to the workshop experiment
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
# MAGIC ---
# MAGIC ## 8. Lakehouse monitor

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
# MAGIC ---
# MAGIC ## 9. MLflow experiment
# MAGIC
# MAGIC Delete the workshop experiment so its **path is freed**. This is the step the reset *script*
# MAGIC omits — and it matters: a UC trace destination can only bind to a *trace-free* experiment, so
# MAGIC an existing experiment (with traces) silently forces Module 0 back onto the default trace store.
# MAGIC Deleting it here lets the next Module 0 run create a fresh experiment and bind UC trace storage.
# MAGIC
# MAGIC > The delete is recoverable from the workspace **Trash** (~30 days); it does not occupy the
# MAGIC > path while trashed, so re-running Module 0 creates a brand-new experiment id.

# COMMAND ----------

import mlflow

_exp_client = MlflowClient()  # workspace tracking store (not UC registry)
_exp = _exp_client.get_experiment_by_name(EXPERIMENT_PATH)
if _exp is not None:
    try:
        _exp_client.delete_experiment(_exp.experiment_id)
        print(f"  Deleted experiment: {EXPERIMENT_PATH} (id={_exp.experiment_id}) — path freed for a clean UC-trace re-bind")
    except Exception as exc:
        print(f"  Skipped experiment delete: {exc}")
else:
    print(f"  No experiment at {EXPERIMENT_PATH} — nothing to delete")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 10. UC schema (cascade)
# MAGIC
# MAGIC `DROP SCHEMA ... CASCADE` removes every table/view/feature-table inside the per-user schema in
# MAGIC one shot — including the `mlflow_traces_*` UC trace tables and `workshop_state`.

# COMMAND ----------

try:
    spark.sql(f"DROP SCHEMA IF EXISTS {FULL_SCHEMA} CASCADE")
    print(f"  Dropped schema: {FULL_SCHEMA}")
except Exception as exc:
    print(f"  Schema drop failed: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Done
# MAGIC
# MAGIC Every workshop resource for this user has been torn down. To start fresh:
# MAGIC
# MAGIC 1. Open `setup/00_setup_and_synthetic_data.py` and `Run All` (re-creates the schema + experiment,
# MAGIC    and — with the trace-storage previews enabled — binds UC trace storage before the first trace).
# MAGIC 2. Continue through Modules 1 → 10, or just re-run the `MLFlow Workshop e2e job`.
# MAGIC
# MAGIC **Note** — this module does **not** delete the `MLFlow Workshop e2e job` (Workflows) or this Git
# MAGIC folder; those are reusable infrastructure, not per-run data. Remove the job from the Workflows UI
# MAGIC (or `w.jobs.delete(job_id=...)`) if you want it gone too.
