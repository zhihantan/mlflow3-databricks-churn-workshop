# Databricks notebook source
# MAGIC %md
# MAGIC # Module 05 — Production Monitoring with MLflow + scipy
# MAGIC ### Catch drift before it silently degrades retention targeting
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > **TL;DR** — Simulate a 2-window inference table where window 2 has deliberately shifted feature distributions. Compute drift with `scipy.stats` (KS for numerics, χ² for categoricals), evaluate model-performance shift with `mlflow.evaluate(model_type="classifier")` per window, persist results to a queryable Delta drift table + MLflow time-series metrics. A SQL Alert closes the loop.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC In production, the model you deployed in Module 4 will eventually face data drift — payment behaviors shift, support-ticket volumes spike, new countries onboard. This module builds a **simulated** inference table with synthetic drift, then computes drift metrics with `scipy.stats` and tracks them as a time series in MLflow + a Delta `churn_drift_metrics` table.
# MAGIC
# MAGIC > **Note**: The original workshop used `databricks.lakehouse_monitoring` (legacy package, removed from PyPI) to auto-generate profile + drift tables. That capability now lives on the Databricks SDK as `WorkspaceClient.quality_monitors` and is the forward-looking path for full production monitoring. We compute the equivalent metrics manually here to keep the dependency footprint small and the math visible.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Build a **simulated** inference table with two time windows where window 2 has a deliberately drifted `payment_failures_60d` distribution.
# MAGIC - Compute per-feature drift metrics — Kolmogorov–Smirnov for numerics, chi-squared for categoricals — and a window-over-window mean shift.
# MAGIC - Persist drift metrics to a queryable Delta table + log them to MLflow for time-series tracking.
# MAGIC - Use `mlflow.evaluate(model_type="classifier")` per window to detect *prediction* drift (not just input drift) — model performance shift over time.
# MAGIC
# MAGIC **Databricks features showcased**
# MAGIC
# MAGIC - **Delta inference table** with Change Data Feed enabled — every prediction the model makes is durable, queryable, and time-travel-able. The same table backs drift detection, retraining label backfills, and audit trails.
# MAGIC - **`mlflow.evaluate(model_type="classifier")` per window** — turns "did model performance shift?" into a 4-line cell, with metrics auto-logged into MLflow for direct chart-based comparison across windows / model versions / dates.
# MAGIC - **MLflow as the metric time-series store** — every drift refresh is a new MLflow run with per-feature drift statistics as logged metrics. Open the experiment, chart `drift_p_value__payment_failures_60d`, and you have a free SRE-grade dashboard for free.
# MAGIC - **Delta + Databricks SQL Alerts** — the persisted `<schema>.churn_drift_metrics` table is queryable from SQL Warehouses, Lakeview dashboards, and SQL Alerts. Setting an alert ("email me when `features_with_drift > 0`") is a 30-second SQL config, no Airflow/PagerDuty plumbing required.
# MAGIC - **(Forward-looking) Lakehouse Monitoring** — `WorkspaceClient.quality_monitors` on the Databricks SDK is the fully-managed path. It auto-generates the same drift Delta tables + a dashboard. We hand-roll the math here so the *concept* is visible, but for production you'd flip to the managed surface.
# MAGIC
# MAGIC **Why this matters for insurtech**
# MAGIC
# MAGIC bolttech's customer base spans 14 APAC + EMEA markets where macro events shift feature distributions overnight — a regional payment-processor outage drives `payment_failures_60d` up; a new pricing tier rollout changes `plan_tier` distribution; a viral social-media incident spikes `support_ticket_count_30d`. Without drift detection the churn model silently degrades, retention campaigns target the wrong customers, and the discovery happens only after a quarter of poor renewal numbers. With a daily-refreshed drift Delta + MLflow time series + a SQL Alert, the data team catches macro shifts in hours and triggers a retraining pipeline before customer-experience damage compounds.
# MAGIC
# MAGIC **Why simulated and not real endpoint traffic?** The Module 4 endpoint only has whatever traffic our 3 sample predictions generated. Drift detection needs population over time. Simulating gives a deterministic, didactically clean drift signal in seconds instead of needing participants to send hundreds of predictions.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 0, 1, 4 have been run.
# MAGIC
# MAGIC **Expected runtime**: ~2-3 minutes (everything runs synchronously; no async monitor refresh to wait for).
# MAGIC
# MAGIC **Compute**: Serverless or DBR 17.3 LTS ML. `scipy` is preinstalled on both.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-sdk>=0.40"
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
    INFERENCE_TABLE,
    SNAPSHOTS_TABLE,
    SNAPSHOT_DATE_STR,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Recover the deployed model version from workshop state

# COMMAND ----------

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"
state_rows = {r["key"]: r["value"] for r in spark.table(STATE_TABLE).select("key", "value").collect()}
champion_version = state_rows.get("churn_champion_version", "1")
print(f"Champion model version: {champion_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
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
# MAGIC ---
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
# MAGIC ---
# MAGIC ## 5. Compute per-feature drift metrics (window 2 vs window 1)
# MAGIC
# MAGIC For each feature we compute window-over-window drift relative to window 1 as the baseline:
# MAGIC - **Numeric features** → two-sample Kolmogorov–Smirnov test. The KS statistic measures the maximum vertical distance between two ECDFs; the p-value tells us whether the distributions are statistically distinguishable.
# MAGIC - **Categorical features** → chi-squared test on the contingency table of category counts.
# MAGIC
# MAGIC We also compute a plain mean-shift percentage for numerics — easier to read than a KS statistic for stakeholders.
# MAGIC
# MAGIC Ref: https://docs.scipy.org/doc/scipy/reference/stats.html

# COMMAND ----------

import mlflow
from scipy import stats

NUMERIC_FEATURES = ["payment_failures_60d", "pending_claims_90d"]
CATEGORICAL_FEATURES = ["country", "plan_tier"]
ALPHA = 0.05  # drift-detected threshold on p-value

window_1 = inference_pdf[inference_pdf["inference_ts"] < pd.Timestamp(WINDOW_2_START)]
window_2 = inference_pdf[inference_pdf["inference_ts"] >= pd.Timestamp(WINDOW_2_START)]
print(f"Window 1 (baseline): {len(window_1):,} rows")
print(f"Window 2 (drifted):  {len(window_2):,} rows\n")

drift_rows: list[dict] = []

for col in NUMERIC_FEATURES:
    w1 = window_1[col].dropna().astype(float)
    w2 = window_2[col].dropna().astype(float)
    ks_stat, ks_p = stats.ks_2samp(w1, w2)
    mean_shift_pct = (w2.mean() - w1.mean()) / w1.mean() * 100 if w1.mean() else 0.0
    drift_rows.append({
        "feature": col,
        "feature_type": "numeric",
        "w1_mean": float(w1.mean()),
        "w1_std": float(w1.std()),
        "w2_mean": float(w2.mean()),
        "w2_std": float(w2.std()),
        "mean_shift_pct": float(mean_shift_pct),
        "test_statistic": float(ks_stat),
        "p_value": float(ks_p),
        "test_name": "ks_2samp",
        "drift_detected": bool(ks_p < ALPHA),
    })

for col in CATEGORICAL_FEATURES:
    all_categories = sorted(set(window_1[col].dropna()) | set(window_2[col].dropna()))
    w1_freq = [int((window_1[col] == c).sum()) for c in all_categories]
    w2_freq = [int((window_2[col] == c).sum()) for c in all_categories]
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency([w1_freq, w2_freq])
    drift_rows.append({
        "feature": col,
        "feature_type": "categorical",
        "w1_mean": None,
        "w1_std": None,
        "w2_mean": None,
        "w2_std": None,
        "mean_shift_pct": None,
        "test_statistic": float(chi2_stat),
        "p_value": float(chi2_p),
        "test_name": "chi2_contingency",
        "drift_detected": bool(chi2_p < ALPHA),
    })

drift_pdf = pd.DataFrame(drift_rows)
print(drift_pdf.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Persist drift metrics + log to MLflow
# MAGIC
# MAGIC Two destinations:
# MAGIC - **Delta table** `<schema>.churn_drift_metrics` — queryable from SQL, dashboards, alerts. Re-runnable: each run overwrites the table.
# MAGIC - **MLflow run** — logs the drift test statistics as metrics + tags so you can compare across model versions and across time. In a real production setup you'd schedule this notebook as a Job and the MLflow run history becomes your time series.

# COMMAND ----------

DRIFT_TABLE = f"{FULL_SCHEMA}.churn_drift_metrics"

drift_sdf = spark.createDataFrame(drift_pdf)
(
    drift_sdf.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(DRIFT_TABLE)
)
print(f"Wrote {DRIFT_TABLE}: {spark.table(DRIFT_TABLE).count()} rows")

with mlflow.start_run(run_name="churn_drift_window2_vs_window1") as drift_run:
    mlflow.log_param("baseline_window", "window_1")
    mlflow.log_param("comparison_window", "window_2")
    mlflow.log_param("champion_version", champion_version)
    mlflow.log_param("baseline_rows", len(window_1))
    mlflow.log_param("comparison_rows", len(window_2))
    mlflow.log_param("alpha", ALPHA)

    for row in drift_rows:
        feat = row["feature"]
        mlflow.log_metric(f"drift_test_statistic__{feat}", row["test_statistic"])
        mlflow.log_metric(f"drift_p_value__{feat}", row["p_value"])
        mlflow.log_metric(f"drift_detected__{feat}", 1.0 if row["drift_detected"] else 0.0)
        if row["mean_shift_pct"] is not None:
            mlflow.log_metric(f"mean_shift_pct__{feat}", row["mean_shift_pct"])

    drift_count = int(drift_pdf["drift_detected"].sum())
    mlflow.log_metric("features_with_drift", drift_count)
    mlflow.set_tag("drift_summary", f"{drift_count}/{len(drift_pdf)} features drifted at α={ALPHA}")

print(f"\nMLflow run: {drift_run.info.run_id}")
print(f"Features with drift detected: {drift_count}/{len(drift_pdf)}")
display(spark.table(DRIFT_TABLE).orderBy(F.col("p_value").asc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7. Prediction-level drift via `mlflow.evaluate()`
# MAGIC
# MAGIC Input drift is only half the story — what we really care about is whether *model performance* has shifted. We use `mlflow.evaluate(model_type="classifier")` once per window to compute classification metrics (accuracy, F1, log loss, AUC) on the predictions already in the inference table, then log them as MLflow metrics so window 1 vs window 2 are directly comparable in the MLflow UI.
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.evaluate

# COMMAND ----------

per_window_metrics: list[dict] = []

for window_label, window_df in [("window_1_baseline", window_1), ("window_2_drifted", window_2)]:
    eval_df = window_df[["actual_churned", "predicted_churn_label"]].rename(
        columns={"actual_churned": "target", "predicted_churn_label": "prediction"}
    )

    with mlflow.start_run(run_name=f"churn_eval_{window_label}") as eval_run:
        mlflow.log_param("window", window_label)
        mlflow.log_param("champion_version", champion_version)
        mlflow.log_param("rows", len(eval_df))

        result = mlflow.evaluate(
            data=eval_df,
            model_type="classifier",
            targets="target",
            predictions="prediction",
        )
        flat_metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
        flat_metrics["window"] = window_label
        flat_metrics["run_id"] = eval_run.info.run_id
        per_window_metrics.append(flat_metrics)

        print(f"\n{window_label}: run_id={eval_run.info.run_id}")
        for k in ("accuracy_score", "f1_score", "log_loss", "roc_auc"):
            if k in flat_metrics:
                print(f"  {k:>20s}: {flat_metrics[k]:.4f}")

# Show window-over-window delta so the drift in performance is obvious at a glance
if len(per_window_metrics) == 2:
    w1m, w2m = per_window_metrics[0], per_window_metrics[1]
    print("\nWindow-over-window deltas:")
    for k in sorted(set(w1m) & set(w2m)):
        if isinstance(w1m[k], float) and k not in ("run_id",):
            delta = w2m[k] - w1m[k]
            print(f"  Δ {k:>22s}: {delta:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 8. Drift query against the persisted Delta table
# MAGIC
# MAGIC Same idea as the Lakehouse-Monitor output — but on a Delta table you wrote yourself. Plug this into a Databricks SQL alert (or a Job that emails when `features_with_drift > 0`) for production-grade retraining triggers.

# COMMAND ----------

# Features ranked by statistical significance of drift (lowest p-value first).
# Plug this query into a Databricks SQL Alert against the same table to email/Slack
# when any feature breaches your significance threshold.
display(
    spark.sql(f"""
        SELECT
            feature,
            feature_type,
            ROUND(test_statistic, 4)  AS test_statistic,
            ROUND(p_value, 6)         AS p_value,
            ROUND(mean_shift_pct, 2)  AS mean_shift_pct,
            drift_detected
        FROM {DRIFT_TABLE}
        ORDER BY p_value ASC
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just learned**
# MAGIC
# MAGIC - The shape of an inference table: predictions, labels, timestamps, model_id, original features.
# MAGIC - Two complementary drift checks:
# MAGIC   - **Input drift** — Kolmogorov–Smirnov for numerics, chi-squared for categoricals, computed with `scipy.stats`.
# MAGIC   - **Prediction drift / performance shift** — `mlflow.evaluate(model_type="classifier")` per window.
# MAGIC - Persisting drift as a Delta table makes it queryable from SQL and pluggable into Databricks SQL alerts; logging to MLflow gives you a per-run history you can chart in the MLflow UI.
# MAGIC - For full production-grade monitoring (auto-generated dashboards, slicing, async refresh) the forward-looking path is `WorkspaceClient.quality_monitors` on the Databricks SDK. The math you saw here is exactly what that service computes under the hood.
# MAGIC
# MAGIC **What you'd build without Databricks**
# MAGIC
# MAGIC | Concern | DIY stack | Databricks-native |
# MAGIC | --- | --- | --- |
# MAGIC | Inference logging | Custom service writing to S3 / Postgres; manual schema management | Delta inference table (1 SQL `saveAsTable`) with CDF + time travel |
# MAGIC | Drift compute | Standalone Python job in Airflow + custom alerting | `scipy.stats` cell + MLflow log_metric, or fully-managed Lakehouse Monitoring |
# MAGIC | Drift dashboards | Grafana / Tableau / custom React | MLflow Experiment UI auto-charts logged metrics; Lakeview dashboards over the Delta table |
# MAGIC | Alerts on drift breach | PagerDuty + custom integrations | Databricks SQL Alert ("when `features_with_drift > 0` send Slack") in 30 seconds |
# MAGIC | Retraining trigger | Custom orchestration to coordinate detection → training → deploy | Same Databricks Job DAG chains M5 → re-run M2/M3 → M4 redeploy |
# MAGIC
# MAGIC **How this composes in production**
# MAGIC
# MAGIC Schedule this notebook as a daily Databricks Job task that depends on M4's batch_predictions table refresh. Add a SQL Alert on the drift table for an automated "model needs retraining" signal. When the alert fires, the next-job-in-chain re-runs the training pipeline (Modules 2-4), promotes the new model to `@champion`, and the inference table picks up the new model version automatically — closing the MLOps feedback loop without any custom orchestration.
# MAGIC
# MAGIC **What's next — Module 6: Tracing & Prompt Registry**
# MAGIC
# MAGIC Module 6 opens the **GenAI half of the workshop**. We'll make a call against the Foundation Model APIs with MLflow Tracing on, register a prompt template in the Prompt Registry, and start the Vector Search endpoint provisioning that Module 7 will use. Open `modules/06_tracing_and_prompts/06_tracing_and_prompts.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
# MAGIC - [scipy.stats.chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
# MAGIC - [mlflow.evaluate](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.evaluate)
# MAGIC - [Databricks Lakehouse Monitoring (forward-looking SDK path)](https://docs.databricks.com/aws/en/lakehouse-monitoring/)
# MAGIC - [Databricks SQL Alerts](https://docs.databricks.com/aws/en/sql/user/alerts/)
