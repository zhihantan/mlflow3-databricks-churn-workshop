# Databricks notebook source
# MAGIC %md
# MAGIC # Module 05 — Production Monitoring with Lakehouse Monitoring
# MAGIC
# MAGIC In production, the model you deployed in Module 4 will eventually face data drift — payment behaviors shift, support-ticket volumes spike, new countries onboard. Lakehouse Monitoring continuously profiles your inference table and surfaces drift relative to a baseline, so you can decide when to retrain.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Build a **simulated** inference table with two time windows where window 2 has a deliberately drifted `payment_failures_60d` distribution.
# MAGIC - Create a Lakehouse Monitor with `InferenceLog` profile type pointing at that table.
# MAGIC - See the schema of the auto-generated profile + drift metric tables, and run a sample drift query.
# MAGIC
# MAGIC **Why simulated and not real endpoint traffic?** The Module 4 endpoint only has whatever traffic our 3 sample predictions generated. Lakehouse Monitoring needs population over time to detect drift. Simulating gives a deterministic, didactically clean drift signal in seconds instead of needing participants to send hundreds of predictions.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 0, 1, 4 have been run.
# MAGIC
# MAGIC **Expected runtime**: ~3-4 minutes (monitor creation is the slow part; metric computation runs async and may not finish during the workshop — that's fine, the schema and the *setup pattern* are the lesson).
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-sdk>=0.40" \
# MAGIC   "databricks-lakehouse-monitoring"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
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
    INFERENCE_TABLE,
    SNAPSHOTS_TABLE,
    SNAPSHOT_DATE_STR,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Recover the deployed model version from workshop state

# COMMAND ----------

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"
state_rows = {r["key"]: r["value"] for r in spark.table(STATE_TABLE).select("key", "value").collect()}
champion_version = state_rows.get("churn_champion_version", "1")
print(f"Champion model version: {champion_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build a simulated 2-window inference table
# MAGIC
# MAGIC We synthesize 7 days of inferences per window. Window 1 (the baseline window, days 1-7) is drawn directly from the feature table. Window 2 (the drifted window, days 8-14) takes the same sample but multiplies `payment_failures_60d` by 2 and `pending_claims_90d` by 1.5 — simulating a real-world shift where payment infrastructure problems or a backlog of claims cause the distribution to drift.

# COMMAND ----------

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

SNAPSHOT_DATE = date.fromisoformat(SNAPSHOT_DATE_STR)
WINDOW_1_START = SNAPSHOT_DATE
WINDOW_2_START = SNAPSHOT_DATE + timedelta(days=7)

# Sample 500 rows per day for 14 days → 7000 rows total
features_pdf = spark.table(f"{FULL_SCHEMA}.customer_churn_features").toPandas()
labels_pdf = spark.table(SNAPSHOTS_TABLE).toPandas()
labels_lookup = dict(zip(labels_pdf["customer_id"], labels_pdf["churned"]))

rng = np.random.default_rng(seed=42)
N_PER_DAY = 500
window_rows: list[dict] = []

for day_offset in range(14):
    inference_ts = pd.Timestamp(WINDOW_1_START + timedelta(days=day_offset, hours=int(rng.integers(8, 20))))
    sample = features_pdf.sample(n=N_PER_DAY, random_state=int(rng.integers(0, 10_000)), replace=True).copy()
    # Apply drift in window 2
    if day_offset >= 7:
        sample["payment_failures_60d"] = sample["payment_failures_60d"] * 2
        sample["pending_claims_90d"] = (sample["pending_claims_90d"] * 1.5).round().astype(int)
    sample["inference_ts"] = inference_ts
    sample["model_version"] = champion_version
    # Mock predictions — a simple logit using the drifted features so the prediction
    # distribution itself shifts and the monitor can pick that up too.
    logit = -2.2 + 0.55 * sample["payment_failures_60d"] + 0.35 * sample["pending_claims_90d"] + 0.2 * (sample["plan_tier"] == "basic").astype(int)
    prob = 1.0 / (1.0 + np.exp(-logit))
    sample["predicted_churn_proba"] = prob.clip(0, 1)
    sample["predicted_churn_label"] = (prob >= 0.5).astype(int)
    sample["actual_churned"] = sample["customer_id"].map(labels_lookup).fillna(0).astype(int)
    window_rows.append(sample)

inference_pdf = pd.concat(window_rows, ignore_index=True)
print(f"Simulated inference table: {len(inference_pdf):,} rows × {len(inference_pdf.columns)} columns")
print(f"Window 1 avg payment_failures_60d: {inference_pdf[inference_pdf['inference_ts'] < pd.Timestamp(WINDOW_2_START)]['payment_failures_60d'].mean():.2f}")
print(f"Window 2 avg payment_failures_60d: {inference_pdf[inference_pdf['inference_ts'] >= pd.Timestamp(WINDOW_2_START)]['payment_failures_60d'].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write the inference table to Delta

# COMMAND ----------

inference_sdf = spark.createDataFrame(inference_pdf)
(
    inference_sdf.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(INFERENCE_TABLE)
)
print(f"Wrote {INFERENCE_TABLE}: {spark.table(INFERENCE_TABLE).count():,} rows")
display(spark.table(INFERENCE_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create the Lakehouse Monitor
# MAGIC
# MAGIC We use the legacy `databricks.lakehouse_monitoring` API here — it's stable, well-documented, and ships on DBR ML LTS. The newer `WorkspaceClient.data_quality` SDK surface (see VERIFICATION.md) is the forward-looking path; switch when it stabilizes.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/lakehouse-monitoring/create-monitor-api
# MAGIC
# MAGIC Key choices:
# MAGIC - `profile_type=InferenceLog(...)` — profile_type telling the monitor this is ML inferences (not arbitrary time-series).
# MAGIC - `problem_type="classification"` — the type of ML problem being monitored.
# MAGIC - `prediction_col`, `label_col`, `timestamp_col`, `model_id_col` — required InferenceLog columns.
# MAGIC - `granularities=["1 day"]` — daily aggregation. Workshop simulates 14 days, so we'll see 14 buckets.
# MAGIC - `slicing_exprs=["country", "plan_tier"]` — additionally break out metrics by these dimensions.

# COMMAND ----------

import databricks.lakehouse_monitoring as lm  # type: ignore

# Drop any prior monitor for idempotency
try:
    lm.delete_monitor(table_name=INFERENCE_TABLE)
    print(f"Deleted prior monitor on {INFERENCE_TABLE}")
except Exception:
    pass

monitor_info = lm.create_monitor(
    table_name=INFERENCE_TABLE,
    profile_type=lm.InferenceLog(
        problem_type="classification",
        prediction_col="predicted_churn_label",
        label_col="actual_churned",
        timestamp_col="inference_ts",
        granularities=["1 day"],
        model_id_col="model_version",
    ),
    output_schema_name=FULL_SCHEMA,
    slicing_exprs=["country", "plan_tier"],
)
print(f"Monitor created. Status: {monitor_info.status}")
print(f"Profile metrics table: {monitor_info.profile_metrics_table_name}")
print(f"Drift metrics table:   {monitor_info.drift_metrics_table_name}")
print(f"Dashboard URL (once metrics compute): {monitor_info.dashboard_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Inspect the auto-generated metric tables
# MAGIC
# MAGIC The monitor creates two empty tables immediately; metric rows populate asynchronously after the first refresh (~5-15 min on a small table). For the workshop we just show the schemas — the *takeaway* is that you now have versioned, time-bucketed profile and drift metrics queryable as Delta tables, which means you can plug them into alerts, dashboards, Jobs, anywhere.

# COMMAND ----------

profile_metrics_table = monitor_info.profile_metrics_table_name
drift_metrics_table = monitor_info.drift_metrics_table_name

print(f"\n=== Profile metrics table schema: {profile_metrics_table} ===")
spark.table(profile_metrics_table).printSchema()

print(f"\n=== Drift metrics table schema: {drift_metrics_table} ===")
spark.table(drift_metrics_table).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Trigger an initial refresh
# MAGIC
# MAGIC `lm.run_refresh(...)` kicks off metric computation. We submit it but don't wait — refresh takes longer than the workshop budget. The instructor can show populated metrics from a pre-run setup, or participants can come back to this notebook later.

# COMMAND ----------

refresh_info = lm.run_refresh(table_name=INFERENCE_TABLE)
print(f"Refresh submitted: refresh_id={refresh_info.refresh_id}, state={refresh_info.state}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Example drift query (works once metrics populate)
# MAGIC
# MAGIC The query below would surface the drift on `payment_failures_60d` once the monitor refresh completes. For now it will return empty — wait ~10 min after the refresh and re-run, or share an instructor-prepared screenshot.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show drift on payment_failures_60d window-over-window.
# MAGIC -- The monitor populates ${full_schema}.churn_inferences_sim_drift_metrics (or similar)
# MAGIC -- after the first refresh. Replace the table name with `monitor_info.drift_metrics_table_name`
# MAGIC -- (printed above) if the auto-construction below doesn't match your workspace.
# MAGIC SELECT 1 AS placeholder
# MAGIC -- SELECT window, column_name, drift_type, js_distance, wasserstein_distance
# MAGIC -- FROM ${monitor_info.drift_metrics_table_name}
# MAGIC -- WHERE column_name IN ('payment_failures_60d', 'pending_claims_90d', 'predicted_churn_label')
# MAGIC -- ORDER BY window DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just learned**
# MAGIC
# MAGIC - The shape of a Lakehouse Monitoring inference table: predictions, labels, timestamps, model_id, and the original features.
# MAGIC - `lm.create_monitor(profile_type=InferenceLog(...))` for ML inference monitoring with classification problem type.
# MAGIC - The monitor auto-generates two Delta tables — profile metrics + drift metrics — plus a dashboard. You can query them like any other Delta table.
# MAGIC - Slicing via `slicing_exprs` gives you "drift by country" / "drift by plan tier" out of the box.
# MAGIC
# MAGIC **What's next — Module 6: Tracing & Prompt Registry**
# MAGIC
# MAGIC Module 6 opens the **GenAI half of the workshop**. We'll make a call against the Foundation Model APIs with MLflow Tracing on, register a prompt template in the Prompt Registry, and start the Vector Search endpoint provisioning that Module 7 will use. Open `modules/06_tracing_and_prompts/06_tracing_and_prompts.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [Lakehouse Monitoring overview](https://docs.databricks.com/aws/en/lakehouse-monitoring/)
# MAGIC - [Create a monitor via API](https://docs.databricks.com/aws/en/lakehouse-monitoring/create-monitor-api)
# MAGIC - [Monitor output tables](https://docs.databricks.com/aws/en/lakehouse-monitoring/monitor-output)
