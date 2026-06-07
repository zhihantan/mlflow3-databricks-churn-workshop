# Databricks notebook source
# MAGIC %md
# MAGIC # Module 0 — Workshop Setup & Synthetic bolttech Data
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will have:
# MAGIC
# MAGIC - A Unity Catalog **catalog + schema** created for the workshop, scoped to your username so the workshop runs cleanly on a shared workspace.
# MAGIC - Six Delta tables of realistic synthetic insurtech data — customers, policies, claims, payments, support tickets, and a labeled customer snapshot — written to your schema with Change Data Feed enabled.
# MAGIC - A working `workshop_config.py` import path validated for downstream modules.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - A Databricks workspace with **Unity Catalog enabled**.
# MAGIC - Permissions: `CREATE CATALOG` on the metastore (if creating `bolttech_workshop` for the first time) OR `USE CATALOG bolttech_workshop` + `CREATE SCHEMA` on the catalog (if it already exists).
# MAGIC - **Compute:** Serverless base environment, OR a classic cluster running **Databricks Runtime 17.3 LTS ML**.
# MAGIC
# MAGIC **Expected runtime**: ~2-3 minutes (20k customers + supporting tables generated in pandas, written to Delta as a single Spark write per table).
# MAGIC
# MAGIC **What you'll have at the end**: a self-contained synthetic data foundation for the rest of the workshop. No external downloads, no hand-editing of names anywhere — every identifier in every downstream module reads from `config/workshop_config.py`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install pinned dependencies
# MAGIC
# MAGIC DBR 17.3 LTS ML ships MLflow 3.0.1, but several modules in this workshop need MLflow 3.12+ for the current `mlflow.genai` surface (predefined scorers, ResponsesAgent, prompt aliases). We `%pip install` the workshop pins at the top of every module so the run is identical on classic and Serverless ML compute.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "databricks-sdk>=0.40" \
# MAGIC   "databricks-feature-engineering>=0.14"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports & workshop config
# MAGIC
# MAGIC `workshop_config.py` resolves per-user names so a shared workspace can host multiple participants without colliding on catalog / schema / endpoint names.

# COMMAND ----------

# Resolve the repo root so `config.workshop_config` is importable regardless of whether
# this notebook is opened from a Databricks Repo or a Git folder.
#
# `notebookPath()` returns paths like:
#   /Repos/<user>/<repo>/setup/00_setup_and_synthetic_data
#   /Workspace/Users/<user>/<repo>/setup/00_setup_and_synthetic_data
#   /Users/<user>/<repo>/setup/00_setup_and_synthetic_data
# The filesystem requires a leading `/Workspace` prefix to be importable via sys.path,
# so we add it if missing.
import os
import sys

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(_nb_path))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)
print(f"Added to sys.path: {_repo_root}")

from config.workshop_config import (  # noqa: E402
    CATALOG,
    SCHEMA,
    FULL_SCHEMA,
    CUSTOMERS_TABLE,
    POLICIES_TABLE,
    CLAIMS_TABLE,
    PAYMENTS_TABLE,
    TICKETS_TABLE,
    SNAPSHOTS_TABLE,
    EXPERIMENT_PATH,
    MONITORING_WAREHOUSE_ID,
    MONITORING_ENABLED,
    TRACE_TABLE_PREFIX,
    SYNTHETIC_SEED,
    N_CUSTOMERS,
    N_TICKETS,
    SNAPSHOT_DATE_STR,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the catalog + schema (idempotent)
# MAGIC
# MAGIC `CREATE ... IF NOT EXISTS` makes this safe to re-run. If you don't have `CREATE CATALOG` on the metastore, ask your workspace admin to run the first statement once; the rest of the workshop only needs `USE CATALOG bolttech_workshop` + `CREATE SCHEMA`.

# COMMAND ----------

# Create the catalog if you have CREATE CATALOG on the metastore. If you're reusing an
# existing catalog (e.g. a customer-provided one set via WORKSHOP_CATALOG) and lack that
# permission, this is a best-effort no-op — the per-user schema below is all the workshop
# needs. This keeps the same deploy script runnable in any customer workspace.
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as exc:
    print(f"  CREATE CATALOG skipped — reusing existing catalog '{CATALOG}': {exc}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Using {FULL_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3b. Enable Unity Catalog trace storage + production monitoring (on by default)
# MAGIC
# MAGIC The GenAI experiment's **Overview** dashboards (Usage / Quality / Tool calls) are powered by
# MAGIC **Unity Catalog trace storage**, not the default control-plane trace store. To light them up,
# MAGIC the experiment must be created **UC-backed before its first trace** — which is exactly why this
# MAGIC runs here in Module 0, before any module logs a trace.
# MAGIC
# MAGIC This is now **on by default** — `config/workshop_config.py` auto-resolves a SQL warehouse the
# MAGIC runner can use (a pinned `MONITORING_WAREHOUSE_ID` env var wins if set; otherwise it picks a
# MAGIC running serverless warehouse from the workspace). No pre-configuration needed. Module 9 §9's
# MAGIC scheduled scorers then feed the **Quality** tab and these traces feed **Usage**.
# MAGIC
# MAGIC It self-disables only when no warehouse is usable (none visible / no permission / off Databricks),
# MAGIC or when you set `WORKSHOP_DISABLE_MONITORING=1`. In that case the workshop still runs end-to-end on
# MAGIC the default trace store — the **Traces tab** works, only the aggregate Overview charts stay empty.
# MAGIC
# MAGIC This step is **self-verifying**: after binding it emits a probe trace and polls the UC spans
# MAGIC table to confirm writes actually route to UC, printing a loud PASS/WARN and persisting the
# MAGIC outcome to `<schema>.uc_trace_diagnostics`. The bind error is never silently swallowed.
# MAGIC
# MAGIC Prereqs: a SQL warehouse the runner can use + the workspace's trace-storage preview features on.
# MAGIC Ref: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog

# COMMAND ----------

# Ref: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
if not MONITORING_ENABLED:
    print(
        "No usable SQL warehouse resolved — running on the default trace store.\n"
        "Monitoring is on by default but self-disables when the auto-resolver finds no warehouse\n"
        "this principal can use (or WORKSHOP_DISABLE_MONITORING=1 is set). The workshop runs\n"
        "normally (Traces tab works; the aggregate Overview dashboards stay empty). To force a\n"
        "specific warehouse, set the MONITORING_WAREHOUSE_ID env var."
    )
else:
    print(f"Monitoring on — auto-resolved SQL warehouse: {MONITORING_WAREHOUSE_ID}")
    import time
    import traceback
    import mlflow
    from pyspark.sql import Row, functions as F

    mlflow.set_tracking_uri("databricks")

    SPANS_TABLE = f"{FULL_SCHEMA}.{TRACE_TABLE_PREFIX}_otel_spans"
    DIAG_TABLE = f"{FULL_SCHEMA}.uc_trace_diagnostics"
    _diag = {
        "bind_ok": False,
        "bind_error": "",
        "probe_landed": False,
        "spans_before": -1,
        "spans_after": -1,
        "monitoring_warehouse": MONITORING_WAREHOUSE_ID,
    }

    # --- Bind the experiment to UC trace storage --------------------------------------------
    # A UC destination can ONLY attach to an experiment with ZERO traces, so this runs before
    # Module 6 logs the first trace. We deliberately do NOT swallow failures: a silently swallowed
    # bind error is exactly what leaves the Overview/Usage dashboards empty while the Traces tab
    # still works (traces fall back to the default control-plane store).
    try:
        from mlflow.entities.trace_location import UnityCatalog

        _exp = mlflow.set_experiment(
            experiment_name=EXPERIMENT_PATH,
            trace_location=UnityCatalog(
                catalog_name=CATALOG,
                schema_name=SCHEMA,
                table_prefix=TRACE_TABLE_PREFIX,
            ),
        )
        _diag["bind_ok"] = True
        print(f"UC trace storage bind call returned OK: experiment {_exp.experiment_id} -> {FULL_SCHEMA}.{TRACE_TABLE_PREFIX}_*")
    except Exception as exc:
        _diag["bind_error"] = repr(exc)
        print("\n" + "!" * 80)
        print("UC TRACE STORAGE BIND FAILED — traces will fall back to the DEFAULT store and the")
        print("Overview / Usage dashboards will stay EMPTY. Full error (NOT swallowed):")
        print("!" * 80)
        print(traceback.format_exc())
        print("!" * 80 + "\n")
        mlflow.set_experiment(EXPERIMENT_PATH)  # continue on default store rather than fail the run

    # --- Point the monitoring job at the SQL warehouse --------------------------------------
    try:
        from mlflow.tracing import set_databricks_monitoring_sql_warehouse_id

        set_databricks_monitoring_sql_warehouse_id(sql_warehouse_id=MONITORING_WAREHOUSE_ID)
        print(f"Monitoring SQL warehouse set: {MONITORING_WAREHOUSE_ID}")
    except Exception:
        print("Could not set monitoring SQL warehouse:\n" + traceback.format_exc())

    # --- VERIFY writes actually route to UC -------------------------------------------------
    # The bind returning OK is necessary but NOT sufficient — the real test is whether a freshly
    # emitted trace lands in the UC OTel spans Delta table. Emit a probe span and poll the table.
    if _diag["bind_ok"]:
        def _spans_count():
            try:
                return spark.table(SPANS_TABLE).count()
            except Exception:
                return None

        before = _spans_count()
        _diag["spans_before"] = before if before is not None else -1

        @mlflow.trace(name="uc_trace_storage_probe")
        def _uc_trace_probe():
            return "ok"

        _uc_trace_probe()

        after, _waited = before, 0
        while _waited < 150:
            after = _spans_count()
            if before is not None and after is not None and after > before:
                _diag["probe_landed"] = True
                break
            time.sleep(15)
            _waited += 15
        _diag["spans_after"] = after if after is not None else -1

        if _diag["probe_landed"]:
            print(
                f"\n✅ UC trace WRITE verified — probe span landed in {SPANS_TABLE} "
                f"(rows {_diag['spans_before']} -> {_diag['spans_after']}). "
                f"Overview/Usage will populate as Modules 6+ emit traces.\n"
            )
        else:
            print("\n" + "?" * 80)
            print(f"UC bind returned OK but the probe span did NOT reach {SPANS_TABLE} within 150s.")
            print("Either UC ingestion is lagging (re-check the table in a few minutes) OR trace")
            print("writes are NOT routing to UC — in which case the Overview/Usage tab stays empty.")
            print(f"spans_before={_diag['spans_before']} spans_after={_diag['spans_after']}")
            print("?" * 80 + "\n")

    # --- Persist the outcome so it's queryable after the run --------------------------------
    # Serverless notebook-task logs aren't always retrievable via the Jobs API, so we durably
    # record the result. Query `<schema>.uc_trace_diagnostics` to see what happened on any run.
    try:
        (
            spark.createDataFrame([Row(**_diag)])
            .withColumn("checked_at", F.current_timestamp())
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(DIAG_TABLE)
        )
        print(f"UC trace diagnostics written to {DIAG_TABLE}: {_diag}")
    except Exception:
        print("Could not persist UC trace diagnostics:\n" + traceback.format_exc())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate synthetic customers
# MAGIC
# MAGIC We seed numpy once (`SYNTHETIC_SEED` in `workshop_config.py`) so every participant gets the same data and every re-run is deterministic. All distributions roughly match bolttech's published footprint (Asia-Pacific heavy customer base, device-protection-led product mix).

# COMMAND ----------

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

rng = np.random.default_rng(SYNTHETIC_SEED)
SNAPSHOT_DATE = date.fromisoformat(SNAPSHOT_DATE_STR)

countries = ["SG", "MY", "ID", "TH", "VN", "PH", "HK", "TW", "AU", "JP", "KR", "IN", "GB", "US"]
country_probs = np.array([0.18, 0.10, 0.12, 0.08, 0.05, 0.06, 0.05, 0.04, 0.07, 0.05, 0.04, 0.06, 0.05, 0.05])
country_probs = country_probs / country_probs.sum()

plan_tiers = ["basic", "plus", "premium"]
plan_probs = [0.55, 0.30, 0.15]

device_types = ["iphone", "android", "tablet", "laptop"]
device_probs = [0.45, 0.40, 0.10, 0.05]

# Customer cohort: signed up between 2024-01-01 and 2026-03-01 (so at snapshot 2026-04-01 they have 1-26 months tenure)
signup_min = date(2024, 1, 1)
signup_max = date(2026, 3, 1)
signup_span_days = (signup_max - signup_min).days

customer_ids = [f"CUST_{i:06d}" for i in range(1, N_CUSTOMERS + 1)]
customers_df = pd.DataFrame(
    {
        "customer_id": customer_ids,
        "country": rng.choice(countries, N_CUSTOMERS, p=country_probs),
        "age": rng.normal(38, 12, N_CUSTOMERS).clip(18, 75).astype(int),
        "plan_tier": rng.choice(plan_tiers, N_CUSTOMERS, p=plan_probs),
        "signup_date": [signup_min + timedelta(days=int(d)) for d in rng.integers(0, signup_span_days, N_CUSTOMERS)],
        "primary_device": rng.choice(device_types, N_CUSTOMERS, p=device_probs),
        "devices_count": rng.integers(1, 5, N_CUSTOMERS),
        "marketing_consent": rng.choice([True, False], N_CUSTOMERS, p=[0.7, 0.3]),
    }
)
customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"])
customers_df["tenure_days_at_snapshot"] = (pd.Timestamp(SNAPSHOT_DATE) - customers_df["signup_date"]).dt.days.astype(int)

print(f"Generated {len(customers_df):,} customers")
display(customers_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate synthetic policies
# MAGIC
# MAGIC Most customers have 1-2 active policies. Products cover bolttech's core lines: device protection, mobile insurance, embedded warranty, and accidental damage.

# COMMAND ----------

products = ["device_protection", "mobile_insurance", "embedded_warranty", "accidental_damage", "screen_replacement"]
product_probs = [0.35, 0.25, 0.15, 0.15, 0.10]

# Each customer gets 1, 2, or 3 policies (most have 1-2)
policies_per_customer = rng.choice([1, 2, 3], N_CUSTOMERS, p=[0.55, 0.35, 0.10])
total_policies = int(policies_per_customer.sum())

# Build per-policy arrays
policy_customer_ids = np.repeat(customers_df["customer_id"].values, policies_per_customer)
policy_signup_dates = np.repeat(customers_df["signup_date"].values, policies_per_customer)
policy_offsets = rng.integers(0, 60, total_policies)  # days after signup before policy started
policy_start_dates = [pd.Timestamp(s).date() + timedelta(days=int(o)) for s, o in zip(policy_signup_dates, policy_offsets)]
policy_term_months = rng.choice([12, 24], total_policies, p=[0.75, 0.25])
policy_end_dates = [s + timedelta(days=int(m * 30)) for s, m in zip(policy_start_dates, policy_term_months)]

# Premium scales with product type + plan tier
plan_lookup = dict(zip(customers_df["customer_id"], customers_df["plan_tier"]))
base_premiums = {"device_protection": 12, "mobile_insurance": 18, "embedded_warranty": 7, "accidental_damage": 14, "screen_replacement": 6}
plan_multiplier = {"basic": 1.0, "plus": 1.6, "premium": 2.4}

policy_products = rng.choice(products, total_policies, p=product_probs)
monthly_premiums = np.array(
    [
        base_premiums[prod] * plan_multiplier[plan_lookup[cid]] * rng.normal(1.0, 0.08)
        for cid, prod in zip(policy_customer_ids, policy_products)
    ]
).round(2).clip(2.0, None)

# Status: active vs lapsed vs cancelled — most are active for an as-of analysis
policy_status = rng.choice(["active", "lapsed", "cancelled"], total_policies, p=[0.85, 0.10, 0.05])

policies_df = pd.DataFrame(
    {
        "policy_id": [f"POL_{i:07d}" for i in range(1, total_policies + 1)],
        "customer_id": policy_customer_ids,
        "product": policy_products,
        "start_date": policy_start_dates,
        "end_date": policy_end_dates,
        "term_months": policy_term_months,
        "monthly_premium": monthly_premiums,
        "status": policy_status,
        "renewal_count": rng.integers(0, 3, total_policies),
    }
)
print(f"Generated {len(policies_df):,} policies across {policies_df['customer_id'].nunique():,} customers")
display(policies_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate synthetic claims
# MAGIC
# MAGIC Claims are sparse — most active policies never have one. We bias claim density slightly higher on policies with higher premiums (the customer paid for more coverage).

# COMMAND ----------

# 25% of policies generate at least one claim; of those, most have 1, some have 2
claim_rate = 0.25
policies_with_claims = rng.random(total_policies) < claim_rate
claims_per_policy = np.where(policies_with_claims, rng.choice([1, 2, 3], total_policies, p=[0.75, 0.20, 0.05]), 0)
total_claims = int(claims_per_policy.sum())

claim_policy_ids = np.repeat(policies_df["policy_id"].values, claims_per_policy)
claim_customer_ids = np.repeat(policies_df["customer_id"].values, claims_per_policy)
claim_products = np.repeat(policies_df["product"].values, claims_per_policy)
claim_start_anchors = np.repeat(pd.to_datetime(policies_df["start_date"]).values, claims_per_policy)

# Claims happen randomly within policy lifetime, capped at snapshot date
claim_offsets = rng.integers(15, 365, total_claims)
claim_dates = [
    min((pd.Timestamp(a).date() + timedelta(days=int(o))), SNAPSHOT_DATE)
    for a, o in zip(claim_start_anchors, claim_offsets)
]
claim_types = rng.choice(["accidental_damage", "theft", "breakdown", "water_damage", "loss"], total_claims, p=[0.40, 0.15, 0.25, 0.12, 0.08])
claim_amounts = (rng.normal(180, 90, total_claims).clip(20, 2000)).round(2)
claim_status_probs = {"accidental_damage": [0.65, 0.20, 0.15], "theft": [0.45, 0.35, 0.20], "breakdown": [0.70, 0.15, 0.15], "water_damage": [0.55, 0.25, 0.20], "loss": [0.40, 0.40, 0.20]}
claim_statuses = [rng.choice(["approved", "rejected", "pending"], p=claim_status_probs[t]) for t in claim_types]
resolution_days = [int(rng.normal(8, 5)) if s != "pending" else None for s in claim_statuses]
resolution_days = [max(1, d) if d is not None else None for d in resolution_days]

claims_df = pd.DataFrame(
    {
        "claim_id": [f"CLM_{i:07d}" for i in range(1, total_claims + 1)],
        "policy_id": claim_policy_ids,
        "customer_id": claim_customer_ids,
        "product": claim_products,
        "claim_date": claim_dates,
        "claim_type": claim_types,
        "claim_amount": claim_amounts,
        "status": claim_statuses,
        "resolution_days": resolution_days,
    }
)
print(f"Generated {len(claims_df):,} claims (claim density: {len(claims_df) / total_policies:.1%} of policies)")
display(claims_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generate synthetic payments
# MAGIC
# MAGIC One payment per policy per month from the policy start date through the snapshot date. We inject payment failures with a small base rate, then heavily up-weight failures for the customers who'll churn (set up in the snapshot step below) — this is what makes `payment_failures_60d` a useful feature in Module 1.

# COMMAND ----------

# First pass: assign each customer a base failure rate
customer_base_fail_rate = rng.beta(2, 30, N_CUSTOMERS)  # most customers around 5-10% failure rate, long tail to 20%+
customer_base_fail_rate_lookup = dict(zip(customers_df["customer_id"], customer_base_fail_rate))

# We'll generate payments in vectorized form: one row per (policy, month) pair.
# Cap payment history to the last 6 months before the snapshot. This keeps the table
# at a workshop-friendly ~180k rows while still giving Module 1 plenty of headroom
# to compute `payment_failures_60d`. A real production payment table would obviously
# go back further — we trade fidelity for runtime here on purpose.
PAYMENT_LOOKBACK_DAYS = 180


def _expand_payments(policy_row):
    """One row per (policy, month) within the lookback window."""
    earliest_global = SNAPSHOT_DATE - timedelta(days=PAYMENT_LOOKBACK_DAYS)
    start = max(pd.Timestamp(policy_row.start_date).date(), earliest_global)
    end_cap = min(pd.Timestamp(policy_row.end_date).date(), SNAPSHOT_DATE)
    pay_dates = []
    cur = start
    while cur <= end_cap:
        pay_dates.append(cur)
        # Advance ~30 days; using a calendar-month-ish offset
        next_month = (cur.replace(day=1) + timedelta(days=32)).replace(day=min(cur.day, 28))
        cur = next_month
    return pay_dates

# Build payment rows as flat columns rather than nested DataFrames for speed
payment_dates: list[date] = []
payment_policy_ids: list[str] = []
payment_customer_ids: list[str] = []
payment_amounts: list[float] = []

for row in policies_df.itertuples(index=False):
    for d in _expand_payments(row):
        payment_dates.append(d)
        payment_policy_ids.append(row.policy_id)
        payment_customer_ids.append(row.customer_id)
        payment_amounts.append(float(row.monthly_premium))

n_payments = len(payment_dates)

# Per-payment failure probability: customer base rate + small policy-specific variance
fail_probs = np.array([customer_base_fail_rate_lookup[c] + rng.normal(0, 0.02) for c in payment_customer_ids]).clip(0.01, 0.5)
payment_statuses = np.where(rng.random(n_payments) < fail_probs, "failed", "success")
# A small share of failures get refunded the next month — modeled simply as a one-time refunded flag
refunded_mask = (payment_statuses == "failed") & (rng.random(n_payments) < 0.1)
payment_statuses = np.where(refunded_mask, "refunded", payment_statuses)
payment_methods = rng.choice(["card", "bank_transfer", "mobile_wallet"], n_payments, p=[0.65, 0.20, 0.15])

payments_df = pd.DataFrame(
    {
        "payment_id": [f"PAY_{i:09d}" for i in range(1, n_payments + 1)],
        "customer_id": payment_customer_ids,
        "policy_id": payment_policy_ids,
        "payment_date": payment_dates,
        "amount": payment_amounts,
        "status": payment_statuses,
        "payment_method": payment_methods,
    }
)
print(f"Generated {len(payments_df):,} payments across {n_payments:,} customer-policy-months")
print(f"Failure rate: {(payments_df['status'] == 'failed').mean():.1%}")
display(payments_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Pre-compute the churn label
# MAGIC
# MAGIC The label `churned` indicates whether the customer will fail to renew / cancel within 30 days of the snapshot. We compute a churn *probability* from observable signals at the snapshot date, then sample. This makes the synthetic data faithful to the features Module 1 will compute — payment failures, claim activity, and ticket sentiment will all carry signal a classifier can learn.

# COMMAND ----------

snapshot_pd = pd.Timestamp(SNAPSHOT_DATE)

# Per-customer aggregates as of snapshot_date
payments_pd = payments_df.copy()
payments_pd["payment_date"] = pd.to_datetime(payments_pd["payment_date"])
payments_60d = payments_pd[
    (payments_pd["payment_date"] >= (snapshot_pd - pd.Timedelta(days=60)))
    & (payments_pd["payment_date"] <= snapshot_pd)
]
failures_60d = payments_60d[payments_60d["status"] == "failed"].groupby("customer_id").size().rename("payment_failures_60d")

claims_pd = claims_df.copy()
claims_pd["claim_date"] = pd.to_datetime(claims_pd["claim_date"])
claims_90d = claims_pd[
    (claims_pd["claim_date"] >= (snapshot_pd - pd.Timedelta(days=90)))
    & (claims_pd["claim_date"] <= snapshot_pd)
]
claims_count_90d = claims_90d.groupby("customer_id").size().rename("claims_count_90d")
pending_claims_90d = claims_90d[claims_90d["status"] == "pending"].groupby("customer_id").size().rename("pending_claims_90d")

# Logistic combination → churn probability
sig = customers_df.set_index("customer_id")[["plan_tier", "tenure_days_at_snapshot"]].copy()
sig["payment_failures_60d"] = failures_60d.reindex(sig.index, fill_value=0)
sig["claims_count_90d"] = claims_count_90d.reindex(sig.index, fill_value=0)
sig["pending_claims_90d"] = pending_claims_90d.reindex(sig.index, fill_value=0)
sig["plan_basic"] = (sig["plan_tier"] == "basic").astype(int)

logit = (
    -2.5
    + 0.55 * sig["payment_failures_60d"]
    + 0.15 * sig["claims_count_90d"]
    + 0.40 * sig["pending_claims_90d"]
    + 0.30 * sig["plan_basic"]
    - 0.0008 * sig["tenure_days_at_snapshot"]
    + rng.normal(0, 0.5, len(sig))
)
churn_prob = 1.0 / (1.0 + np.exp(-logit))
churned = (rng.random(len(sig)) < churn_prob).astype(int)

print(f"Snapshot churn rate: {churned.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Generate synthetic support tickets
# MAGIC
# MAGIC ~500 tickets across five categories (`billing`, `claim_help`, `technical`, `churn_intent`, `general`). Customers who'll churn are over-represented in `churn_intent` and negative-sentiment `billing` tickets — this is the signal Module 7's RAG and Module 8's agent will lean on.

# COMMAND ----------

TICKET_TEMPLATES = {
    "billing": [
        "My monthly premium for the {plan_tier} plan increased from ${old_premium} to ${new_premium} this cycle without notice. Was there a change in coverage? I need an explanation before my next payment.",
        "I've been charged twice for my {product} policy this month. Please refund the duplicate and tell me how this happened.",
        "Why does my payment for {product} keep failing? My card is valid. This is the {nth} time in two months.",
        "Requesting to switch payment method from card to bank transfer for all my active policies. Please confirm timeline and that no payments will be missed.",
        "Bill says I owe ${amount} but I think I cancelled this policy last month. Please confirm status before charging me.",
    ],
    "claim_help": [
        "Claim {claim_id} was submitted on {claim_date}. It's been {ndays} days with no update. My {device_type} is broken and I need this resolved urgently.",
        "Why was my claim for accidental damage on my {device_type} rejected? The policy clearly covers this kind of incident. I want a human review.",
        "Sent in additional documentation for claim {claim_id} two weeks ago. Status still shows pending review. Need an actual person to look at this.",
        "How do I escalate a claim? My {product} claim from {claim_date} has been in pending status for over 30 days now.",
        "Claim {claim_id} was paid out but the amount is ${actual} not the ${expected} I was quoted. There's a discrepancy.",
    ],
    "technical": [
        "Cannot log into the bolttech mobile app on my {device_type}. Password reset emails are not arriving in inbox or spam. Need help accessing my policy.",
        "The app crashes whenever I try to view my policy documents. Tried on iOS and Android both — same behavior.",
        "Cannot upload photos for my claim. The form keeps timing out after 30 seconds even on a fast connection.",
        "Policy renewal page shows 'no active session' even after a fresh login. Tried three browsers.",
        "Where can I download a copy of my policy schedule? The 'documents' tab in the app is empty.",
    ],
    "churn_intent": [
        "I want to cancel my {product} policy effective immediately. Service has been disappointing and I'm switching to a competitor.",
        "Please confirm I can cancel without a fee. I've been a customer for {nmonths} months and feel I'm not getting value.",
        "Considering not renewing my policy at end of term. Three claims this year all took too long. What retention offer do you have?",
        "Planning to cancel due to repeated payment issues. Unless you can offer a different plan I'm leaving.",
        "Customer service has been unhelpful across three previous tickets. I want to cancel my {product} policy and recover the unused portion of my premium.",
    ],
    "general": [
        "Please confirm my next renewal date for my {product} policy is set correctly. The app shows one date but my email confirmation shows another.",
        "How do I add my partner as a beneficiary on my insurance? Couldn't find the option anywhere in the app.",
        "Quick question — does my {product} policy cover loss while travelling internationally?",
        "I updated my address last month. Can you confirm correspondence and renewal notices will go to the new address?",
        "What's the typical claim processing time for {product}? Want to know expectations in case I have to file.",
    ],
}
TICKET_SENTIMENTS = {"billing": ["negative", "negative", "neutral"], "claim_help": ["negative", "neutral", "negative"], "technical": ["neutral", "neutral", "negative"], "churn_intent": ["negative", "negative", "negative"], "general": ["neutral", "positive", "neutral"]}

# Bias ticket assignment towards churned customers for churn_intent + half of billing
churned_ids = sig.index[churned == 1].tolist()
non_churned_ids = sig.index[churned == 0].tolist()

n_per_category = {"billing": 120, "claim_help": 110, "technical": 80, "churn_intent": 110, "general": 80}
assert sum(n_per_category.values()) == N_TICKETS, "Ticket category counts must sum to N_TICKETS"

def _sample_customer_for_category(category: str) -> str:
    """Bias to-churn customers for churn_intent + 60% of billing tickets."""
    if category == "churn_intent":
        return rng.choice(churned_ids) if len(churned_ids) else rng.choice(non_churned_ids)
    if category == "billing" and rng.random() < 0.6 and churned_ids:
        return rng.choice(churned_ids)
    return rng.choice(customers_df["customer_id"].values)

def _fill_template(template: str, customer_id: str) -> str:
    cust = customers_df.set_index("customer_id").loc[customer_id]
    plan = cust["plan_tier"]
    device = cust["primary_device"]
    # Find a policy/claim/payment for this customer if available, else a generic value
    cust_policies = policies_df[policies_df["customer_id"] == customer_id]
    product = cust_policies["product"].iloc[0] if len(cust_policies) else "device_protection"
    cust_claims = claims_df[claims_df["customer_id"] == customer_id]
    claim_id = cust_claims["claim_id"].iloc[0] if len(cust_claims) else f"CLM_{rng.integers(1, total_claims):07d}"
    claim_date_str = (
        cust_claims["claim_date"].iloc[0].isoformat()
        if len(cust_claims)
        else (SNAPSHOT_DATE - timedelta(days=int(rng.integers(15, 90)))).isoformat()
    )
    old_p = round(float(cust_policies["monthly_premium"].iloc[0]) if len(cust_policies) else 12.50, 2)
    new_p = round(old_p * rng.uniform(1.08, 1.25), 2)
    amount = round(rng.uniform(10, 250), 2)
    actual = round(amount * rng.uniform(0.5, 0.9), 2)
    expected = round(amount, 2)
    return template.format(
        plan_tier=plan,
        old_premium=old_p,
        new_premium=new_p,
        product=product.replace("_", " "),
        nth=rng.choice(["second", "third", "fourth"]),
        amount=amount,
        actual=actual,
        expected=expected,
        claim_id=claim_id,
        claim_date=claim_date_str,
        ndays=int(rng.integers(7, 35)),
        device_type=device,
        nmonths=int(rng.integers(6, 30)),
    )

ticket_rows = []
ticket_idx = 0
for category, count in n_per_category.items():
    templates = TICKET_TEMPLATES[category]
    sentiments = TICKET_SENTIMENTS[category]
    for _ in range(count):
        ticket_idx += 1
        cust_id = _sample_customer_for_category(category)
        template = templates[int(rng.integers(0, len(templates)))]
        ticket_rows.append(
            {
                "ticket_id": f"TIC_{ticket_idx:06d}",
                "customer_id": cust_id,
                "created_at": pd.Timestamp(SNAPSHOT_DATE - timedelta(days=int(rng.integers(0, 30)))),
                "category": category,
                "subject": f"[{category}] {template[:50]}…",
                "description": _fill_template(template, cust_id),
                "sentiment": sentiments[int(rng.integers(0, len(sentiments)))],
            }
        )

tickets_df = pd.DataFrame(ticket_rows)
print(f"Generated {len(tickets_df):,} support tickets")
print(tickets_df["category"].value_counts())
display(tickets_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Build the customer snapshot table (the label table)
# MAGIC
# MAGIC One row per customer at the snapshot date with the binary `churned` label. Module 1 will join this against the feature table (via point-in-time `FeatureLookup`) to build the training set.

# COMMAND ----------

snapshots_df = pd.DataFrame(
    {
        "customer_id": sig.index,
        "snapshot_date": SNAPSHOT_DATE,
        "churned": churned,
    }
)
print(f"Snapshot row count: {len(snapshots_df):,} | churn rate: {snapshots_df['churned'].mean():.1%}")
display(snapshots_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Write all six tables to Delta with Change Data Feed enabled
# MAGIC
# MAGIC CDF is required by Module 7's Delta Sync Vector Search index on `support_tickets`; we enable it on all tables for consistency. `mode("overwrite")` makes this notebook idempotent — re-running cleanly replaces the data.

# COMMAND ----------

def _write_delta(df_pd: pd.DataFrame, target: str) -> None:
    """Write a pandas DataFrame to a Delta UC table with CDF enabled."""
    sdf = spark.createDataFrame(df_pd)
    (
        sdf.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.enableChangeDataFeed", "true")
        .saveAsTable(target)
    )
    print(f"  wrote {target}: {sdf.count():,} rows")

print("Writing Delta tables to", FULL_SCHEMA)
_write_delta(customers_df, CUSTOMERS_TABLE)
_write_delta(policies_df, POLICIES_TABLE)
_write_delta(claims_df, CLAIMS_TABLE)
_write_delta(payments_df, PAYMENTS_TABLE)
_write_delta(tickets_df, TICKETS_TABLE)
_write_delta(snapshots_df, SNAPSHOTS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Sanity-check displays
# MAGIC
# MAGIC A quick visual confirmation that the tables landed.

# COMMAND ----------

for table in (CUSTOMERS_TABLE, POLICIES_TABLE, CLAIMS_TABLE, PAYMENTS_TABLE, TICKETS_TABLE, SNAPSHOTS_TABLE):
    count = spark.table(table).count()
    print(f"  {table}: {count:,} rows")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {SNAPSHOTS_TABLE} LIMIT 5"))

# COMMAND ----------

display(spark.sql(f"SELECT description, category, sentiment FROM {TICKETS_TABLE} LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just built**
# MAGIC
# MAGIC - A per-user UC schema `bolttech_workshop.<churn_<user>>` for the rest of the workshop.
# MAGIC - Six Delta tables of seeded synthetic insurtech data — deterministic, idempotent, CDF-enabled.
# MAGIC - A snapshot table at `2026-04-01` with a meaningful churn label (~15-20% positive rate) driven by realistic signals.
# MAGIC
# MAGIC **What's next — Module 1: Feature Engineering in Unity Catalog**
# MAGIC
# MAGIC Module 1 builds a UC feature table on top of these raw tables using `FeatureEngineeringClient`, then assembles a point-in-time training set with `FeatureLookup(timestamp_lookup_key=...)`. Open `modules/01_feature_engineering/01_feature_engineering.py` next.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [Delta Change Data Feed](https://docs.databricks.com/aws/en/delta/delta-change-data-feed)
# MAGIC - [`databricks-feature-engineering`](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc)
