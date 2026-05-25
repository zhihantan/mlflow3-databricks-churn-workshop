# Databricks notebook source
# MAGIC %md
# MAGIC # Module 01 — Feature Engineering in Unity Catalog
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Build a Unity Catalog **feature table** with `primary_keys` + `timeseries_columns` configured for as-of-timestamp lookups.
# MAGIC - Compute eight churn features per customer from the raw Module 0 tables, joined consistently to the snapshot date.
# MAGIC - Assemble a training set via `FeatureLookup(timestamp_lookup_key=...)` — the point-in-time-correct join pattern that prevents target leakage.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Module 0 (`setup/00_setup_and_synthetic_data.py`) has been run.
# MAGIC
# MAGIC **Expected runtime**: ~3-4 minutes (dominated by the feature table create + write).
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install pinned dependencies
# MAGIC
# MAGIC Same pins as Module 0 — `databricks-feature-engineering` 0.14+ for the current UC feature-table API.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-feature-engineering>=0.14"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports & workshop config

# COMMAND ----------

import os
import sys
from datetime import date

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))  # modules/01_../01_...py → repo root
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

from config.workshop_config import (  # noqa: E402
    FULL_SCHEMA,
    CUSTOMERS_TABLE,
    POLICIES_TABLE,
    CLAIMS_TABLE,
    PAYMENTS_TABLE,
    TICKETS_TABLE,
    SNAPSHOTS_TABLE,
    FEATURE_TABLE,
    SNAPSHOT_DATE_STR,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compute features at the snapshot date
# MAGIC
# MAGIC We compute everything as a single Spark DataFrame keyed on `(customer_id, snapshot_date)`. The features split naturally into three groups:
# MAGIC
# MAGIC 1. **Static demographics** from `customers` — country, plan tier, age, primary device.
# MAGIC 2. **Policy-level rollups** from `policies` — tenure, active policy count, average premium.
# MAGIC 3. **Event-driven aggregates** from `claims`, `payments`, `support_tickets` — strictly windowed to *before* the snapshot date so there's **no target leakage**.
# MAGIC
# MAGIC The windowing math:
# MAGIC
# MAGIC | Aggregate | Window |
# MAGIC | --- | --- |
# MAGIC | `claims_count_90d` | claim_date ∈ [snapshot − 90d, snapshot) |
# MAGIC | `claim_amount_sum_90d` | same window |
# MAGIC | `pending_claims_90d` | same window, status = 'pending' |
# MAGIC | `payment_failures_60d` | payment_date ∈ [snapshot − 60d, snapshot) |
# MAGIC | `support_ticket_count_30d` | created_at ∈ [snapshot − 30d, snapshot) |
# MAGIC | `negative_ticket_share_30d` | same window, share with sentiment = 'negative' |

# COMMAND ----------

from pyspark.sql import functions as F

SNAPSHOT_DATE = date.fromisoformat(SNAPSHOT_DATE_STR)
snapshot_lit = F.to_date(F.lit(SNAPSHOT_DATE_STR))

customers = spark.table(CUSTOMERS_TABLE).alias("c")
policies = spark.table(POLICIES_TABLE)
claims = spark.table(CLAIMS_TABLE)
payments = spark.table(PAYMENTS_TABLE)
tickets = spark.table(TICKETS_TABLE)

# Static demographic columns straight from customers
demo = customers.select(
    F.col("customer_id"),
    F.col("country"),
    F.col("plan_tier"),
    F.col("age"),
    F.col("primary_device"),
    F.col("tenure_days_at_snapshot").alias("policy_tenure_days"),
)

# Policy-level rollups (active = end_date > snapshot)
policy_agg = policies.groupBy("customer_id").agg(
    F.sum(F.when(F.col("end_date") > snapshot_lit, 1).otherwise(0)).alias("active_policy_count"),
    F.avg("monthly_premium").alias("avg_premium"),
    F.count("*").alias("total_policy_count"),
)

# Claims windowed [snapshot - 90d, snapshot)
claim_agg = (
    claims.filter(
        (F.col("claim_date") >= F.date_sub(snapshot_lit, 90))
        & (F.col("claim_date") < snapshot_lit)
    )
    .groupBy("customer_id")
    .agg(
        F.count("*").alias("claims_count_90d"),
        F.sum("claim_amount").alias("claim_amount_sum_90d"),
        F.sum(F.when(F.col("status") == "pending", 1).otherwise(0)).alias("pending_claims_90d"),
    )
)

# Payment failures windowed [snapshot - 60d, snapshot)
payment_agg = (
    payments.filter(
        (F.col("payment_date") >= F.date_sub(snapshot_lit, 60))
        & (F.col("payment_date") < snapshot_lit)
    )
    .groupBy("customer_id")
    .agg(
        F.sum(F.when(F.col("status") == "failed", 1).otherwise(0)).alias("payment_failures_60d"),
        F.count("*").alias("payments_count_60d"),
    )
)

# Support tickets windowed [snapshot - 30d, snapshot)
ticket_agg = (
    tickets.filter(
        (F.col("created_at") >= F.date_sub(snapshot_lit, 30))
        & (F.col("created_at") < snapshot_lit)
    )
    .groupBy("customer_id")
    .agg(
        F.count("*").alias("support_ticket_count_30d"),
        F.avg(F.when(F.col("sentiment") == "negative", 1.0).otherwise(0.0)).alias(
            "negative_ticket_share_30d"
        ),
    )
)

# Left-join everything to the demographic base; fill NULL aggregates with 0
features_df = (
    demo
    .join(policy_agg, on="customer_id", how="left")
    .join(claim_agg, on="customer_id", how="left")
    .join(payment_agg, on="customer_id", how="left")
    .join(ticket_agg, on="customer_id", how="left")
    .fillna(0, subset=[
        "active_policy_count", "total_policy_count",
        "claims_count_90d", "claim_amount_sum_90d", "pending_claims_90d",
        "payment_failures_60d", "payments_count_60d",
        "support_ticket_count_30d", "negative_ticket_share_30d",
    ])
    .fillna(0.0, subset=["avg_premium"])
    .withColumn("snapshot_date", snapshot_lit)
)

# Final column ordering — primary keys first, then features
feature_cols = [
    "customer_id", "snapshot_date",
    "country", "plan_tier", "age", "primary_device",
    "policy_tenure_days", "active_policy_count", "total_policy_count", "avg_premium",
    "claims_count_90d", "claim_amount_sum_90d", "pending_claims_90d",
    "payment_failures_60d", "payments_count_60d",
    "support_ticket_count_30d", "negative_ticket_share_30d",
]
features_df = features_df.select(*feature_cols)
features_df.cache()

print(f"Feature DataFrame: {features_df.count():,} rows × {len(feature_cols)} columns")
display(features_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create the UC feature table
# MAGIC
# MAGIC `FeatureEngineeringClient.create_table` is the UC-native way to register a feature table. The crucial parameters:
# MAGIC
# MAGIC - `primary_keys=['customer_id', 'snapshot_date']` — entity + time, jointly unique.
# MAGIC - `timeseries_columns='snapshot_date'` — marks this as a time-series feature table so `FeatureLookup` can do as-of-timestamp joins.
# MAGIC - `schema=features_df.schema` — copy the Spark schema directly so we don't drift.

# COMMAND ----------

# Ref: https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Idempotent: if the table already exists from a prior run, drop and re-create cleanly
# so the schema is always in sync with the current notebook.
try:
    spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")
except Exception as exc:  # pragma: no cover — defensive
    print(f"  (drop skipped: {exc})")

fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["customer_id", "snapshot_date"],
    timeseries_columns="snapshot_date",
    schema=features_df.schema,
    description=(
        "Per-customer, per-snapshot churn features for the MLflow 3 bolttech workshop. "
        "Windowed event-driven aggregates strictly use data from before the snapshot date."
    ),
)
print(f"Created UC feature table: {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write the feature rows
# MAGIC
# MAGIC `mode='merge'` upserts by primary key, so re-running this cell is safe.

# COMMAND ----------

fe.write_table(name=FEATURE_TABLE, df=features_df, mode="merge")
written = spark.table(FEATURE_TABLE).count()
print(f"Wrote {written:,} feature rows to {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Assemble the training set via point-in-time `FeatureLookup`
# MAGIC
# MAGIC `FeatureLookup(timestamp_lookup_key='snapshot_date')` tells the FE client: "for each row in the labels DataFrame, look up the feature row whose `snapshot_date` is the most recent value at or before the label's `snapshot_date`." With a single snapshot date in this workshop the lookup is degenerate (every label finds an exact-match feature row), but the *pattern* is what production point-in-time training requires — and it would scale unchanged if we logged daily snapshots.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup

labels_df = (
    spark.table(SNAPSHOTS_TABLE)
    .select("customer_id", "snapshot_date", "churned")
)
print(f"Label rows: {labels_df.count():,}")

feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key="customer_id",
        timestamp_lookup_key="snapshot_date",
        feature_names=None,  # None → all features in the table
    )
]

training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churned",
    exclude_columns=["customer_id", "snapshot_date"],  # not features, just keys
)

training_df = training_set.load_df()
print(
    f"Training DataFrame: {training_df.count():,} rows × {len(training_df.columns)} columns "
    f"({sum(1 for _ in training_df.columns) - 1} features + 1 label)"
)
display(training_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Materialize a training view for Module 2
# MAGIC
# MAGIC Module 2 will train on this. We write it as a UC table so it survives notebook restarts.

# COMMAND ----------

TRAINING_TABLE = f"{FULL_SCHEMA}.churn_training_set"
# `FeatureEngineeringClient.TrainingSet.load_df()` internally calls `.persist()`
# to cache feature-lookup lineage metadata. Serverless compute blocks PERSIST TABLE
# (`[NOT_SUPPORTED_WITH_SERVERLESS]`). We round-trip through pandas to detach from
# the FE wrapper before the Delta write. Fine for the workshop's ~20k row scale.
training_pdf = training_df.toPandas()
(
    spark.createDataFrame(training_pdf).write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TRAINING_TABLE)
)
print(f"Materialized training set at {TRAINING_TABLE}: {spark.table(TRAINING_TABLE).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sanity-check the feature distribution
# MAGIC
# MAGIC A quick look at the most predictive features confirms the data has signal worth training on.

# COMMAND ----------

display(
    spark.table(TRAINING_TABLE)
    .groupBy("churned")
    .agg(
        F.count("*").alias("n"),
        F.avg("payment_failures_60d").alias("avg_payment_failures_60d"),
        F.avg("claims_count_90d").alias("avg_claims_count_90d"),
        F.avg("pending_claims_90d").alias("avg_pending_claims_90d"),
        F.avg("negative_ticket_share_30d").alias("avg_negative_ticket_share_30d"),
        F.avg("policy_tenure_days").alias("avg_tenure_days"),
    )
    .orderBy("churned")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just built**
# MAGIC
# MAGIC - A UC feature table at `<schema>.customer_churn_features` with `timeseries_columns='snapshot_date'` for point-in-time lookups.
# MAGIC - A materialized training set at `<schema>.churn_training_set` that joins the snapshot labels against features via `FeatureLookup(timestamp_lookup_key=...)`.
# MAGIC - A leakage-free feature pipeline: every event-driven aggregate strictly uses data with timestamp **<** snapshot_date.
# MAGIC
# MAGIC **What's next — Module 2: Experiment Tracking & LoggedModel**
# MAGIC
# MAGIC Module 2 trains a baseline LR and a LightGBM classifier on this training set, demonstrating the MLflow 3 `LoggedModel` entity and the `models:/<model_id>` URI scheme. Open `modules/02_experiment_tracking/02_experiment_tracking.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [Feature Engineering in UC — feature tables](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc)
# MAGIC - [Time-series feature tables / point-in-time joins](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series)
# MAGIC - [Why point-in-time joins prevent target leakage](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series#point-in-time-lookup)
