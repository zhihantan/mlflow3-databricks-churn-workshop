# Databricks notebook source
# MAGIC %md
# MAGIC # Module 02 — Experiment Tracking & the MLflow 3 `LoggedModel`
# MAGIC
# MAGIC This is the **flagship MLflow 3 module** of the workshop. The biggest conceptual change between MLflow 2 and 3 is that **`LoggedModel` is now a first-class entity** — separate from any single run, with its own lifecycle, its own ID, and its own URI scheme (`models:/<model_id>`). Everything else in this workshop builds on that.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Train two churn classifiers (logistic regression baseline + LightGBM) on the Module 1 training set.
# MAGIC - See how `mlflow.autolog()` + `mlflow.<flavor>.log_model(..., name=...)` produces independent `LoggedModel` entities — not run-scoped artifacts.
# MAGIC - Retrieve a model by its `model_id` and load it via the new `models:/<model_id>` URI scheme.
# MAGIC - Understand the three biggest breaking changes between MLflow 2.x and 3.x that participants will hit when porting old code.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 0, 1 have been run.
# MAGIC
# MAGIC **Expected runtime**: ~5 minutes (LR and LightGBM both train in seconds on 20k rows).
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install pinned dependencies
# MAGIC
# MAGIC MLflow 3.12 is critical here — DBR 17.3 LTS ML ships 3.0.1, which has `LoggedModel` but lacks several of the convenience APIs we use later.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "lightgbm>=4.6" \
# MAGIC   "scikit-learn>=1.6"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports & workshop config

# COMMAND ----------

import os
import sys

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

from config.workshop_config import (  # noqa: E402
    FULL_SCHEMA,
    EXPERIMENT_PATH,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set the MLflow experiment
# MAGIC
# MAGIC Per-user experiment under the workspace user folder so participants on a shared workspace don't trample each other's runs.

# COMMAND ----------

import mlflow

# Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load training data + train/test split
# MAGIC
# MAGIC The training set was materialized by Module 1 at `<schema>.churn_training_set`. We pull it into pandas (20k rows fits easily in memory) and split 80/20.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

TRAINING_TABLE = f"{FULL_SCHEMA}.churn_training_set"
df = spark.table(TRAINING_TABLE).toPandas()

CATEGORICAL = ["country", "plan_tier", "primary_device"]
NUMERIC = [
    "age",
    "policy_tenure_days",
    "active_policy_count",
    "total_policy_count",
    "avg_premium",
    "claims_count_90d",
    "claim_amount_sum_90d",
    "pending_claims_90d",
    "payment_failures_60d",
    "payments_count_60d",
    "support_ticket_count_30d",
    "negative_ticket_share_30d",
]
TARGET = "churned"

X = df[CATEGORICAL + NUMERIC].copy()
y = df[TARGET].astype(int)
# Normalize categorical columns to plain object/string dtype so both the LR and the
# LGBM pipelines see the same input contract (no pandas Categorical dtype anywhere).
for col in CATEGORICAL:
    X[col] = X[col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Churn rate (train): {y_train.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Enable autologging
# MAGIC
# MAGIC In MLflow 3, `mlflow.autolog()` defaults `log_traces=True` (most relevant for GenAI; harmless for classic ML) and now produces a `LoggedModel` per training call instead of a run-scoped artifact. We still call explicit `log_model(...)` below to demonstrate the new `name=` parameter (which replaced `artifact_path=`).
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/ml/tracking/autolog/

# COMMAND ----------

mlflow.autolog(log_input_examples=True, log_model_signatures=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train the logistic regression baseline → LoggedModel #1
# MAGIC
# MAGIC A sklearn `Pipeline` with one-hot for categoricals + standard-scaling for numerics + logistic regression. We name the LoggedModel `"logreg_baseline"` via the new `name=` kwarg.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models import infer_signature

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ]
)
lr_pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))])

with mlflow.start_run(run_name="logreg_baseline") as run_lr:
    lr_pipe.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, lr_pipe.predict_proba(X_test)[:, 1])
    mlflow.log_metric("test_auc", test_auc)

    # Explicit signature — required for Unity Catalog model registration.
    # MLflow 3's input_example-driven auto-inference is unreliable with mixed dtypes,
    # so we always build the signature manually with infer_signature.
    sig_input_lr = X_train.head(3)
    signature_lr = infer_signature(sig_input_lr, lr_pipe.predict(sig_input_lr))

    # Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
    # NOTE: `name=` is MLflow 3's replacement for `artifact_path=` from MLflow 2.x.
    lr_logged = mlflow.sklearn.log_model(
        sk_model=lr_pipe,
        name="logreg_baseline",
        input_example=sig_input_lr,
        signature=signature_lr,
    )

print(f"LR run_id      = {run_lr.info.run_id}")
print(f"LR model_id    = {lr_logged.model_id}")
print(f"LR model_uri   = {lr_logged.model_uri}")
print(f"LR test_auc    = {test_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Train the LightGBM model → LoggedModel #2
# MAGIC
# MAGIC We wrap LightGBM in a sklearn `Pipeline` with a `OneHotEncoder` for the categorical
# MAGIC columns. This gives the logged model a portable string-typed input signature (LR
# MAGIC and LGBM share the same shape), avoiding the dtype-mismatch issues that arise when
# MAGIC LightGBM trains on `pandas.Categorical` but serving-time inputs arrive as strings.
# MAGIC
# MAGIC Trade-off: we lose LightGBM's native tree-splits-on-raw-categories optimization,
# MAGIC but for ~20k rows the accuracy delta is negligible and the production-portability
# MAGIC win is significant. `mlflow.lightgbm.log_model` would still work for a bare LGBM
# MAGIC model — we use `mlflow.sklearn.log_model` here because the wrapped Pipeline is a
# MAGIC sklearn estimator.

# COMMAND ----------

import lightgbm as lgb

lgb_pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ]
)
lgb_pipe = Pipeline([
    ("pre", lgb_pre),
    ("clf", lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        objective="binary",
        random_state=42,
        verbosity=-1,
    )),
])

with mlflow.start_run(run_name="lgbm_baseline") as run_lgb:
    lgb_pipe.fit(X_train, y_train)
    test_auc_lgb = roc_auc_score(y_test, lgb_pipe.predict_proba(X_test)[:, 1])
    mlflow.log_metric("test_auc", test_auc_lgb)

    sig_input_lgb = X_train.head(3)
    signature_lgb = infer_signature(sig_input_lgb, lgb_pipe.predict(sig_input_lgb))

    # Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
    # The Pipeline is a sklearn estimator → use the sklearn flavor. MLflow serializes
    # the OneHotEncoder + LightGBM model together inside the logged model.
    lgb_logged = mlflow.sklearn.log_model(
        sk_model=lgb_pipe,
        name="lgbm_baseline",
        input_example=sig_input_lgb,
        signature=signature_lgb,
    )

print(f"LGBM run_id    = {run_lgb.info.run_id}")
print(f"LGBM model_id  = {lgb_logged.model_id}")
print(f"LGBM model_uri = {lgb_logged.model_uri}")
print(f"LGBM test_auc  = {test_auc_lgb:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Inspect the new `LoggedModel` entity
# MAGIC
# MAGIC The killer feature of MLflow 3 is that each logged model has its own URI (`models:/<model_id>`) and its own page in the experiment UI — independent of the run that created it. You can attach metrics, parameters, datasets, and dependencies to a `LoggedModel` directly.
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/ml/mlflow-3/

# COMMAND ----------

# Retrieve the LoggedModel object by its model_id.
# Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html
lgb_model_entity = mlflow.get_logged_model(lgb_logged.model_id)

print("LoggedModel fields:")
print(f"  model_id      = {lgb_model_entity.model_id}")
print(f"  name          = {lgb_model_entity.name}")
print(f"  experiment_id = {lgb_model_entity.experiment_id}")
print(f"  source_run_id = {lgb_model_entity.source_run_id}")
print(f"  status        = {lgb_model_entity.status}")
print(f"  artifact_location = {lgb_model_entity.artifact_location}")
print(f"  creation_timestamp = {lgb_model_entity.creation_timestamp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Load by `models:/<model_id>` URI
# MAGIC
# MAGIC The new URI scheme replaces `runs:/<run_id>/<artifact_path>` from MLflow 2. The two forms below load the *same* model — but only the second one works in MLflow 3 if you didn't capture the run_id (and runs are no longer the unit of model identity).

# COMMAND ----------

# Load via the new model_id URI. Because the model is now a sklearn Pipeline with
# string-typed inputs, no dtype gymnastics are needed at predict time.
loaded = mlflow.pyfunc.load_model(f"models:/{lgb_logged.model_id}")
preds = loaded.predict(X_test.head(5))
print("Predictions on 5 test rows (loaded via models:/<model_id>):")
print(preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. MLflow 2 → 3: the three biggest breaking changes to call out
# MAGIC
# MAGIC | # | Change | Why it matters |
# MAGIC | --- | --- | --- |
# MAGIC | 1 | `mlflow.<flavor>.log_model(artifact_path=...)` → `name=...` | Every code snippet from MLflow 2 tutorials needs editing. `artifact_path` is silently ignored if you forget. |
# MAGIC | 2 | Models are no longer run-scoped artifacts | `runs:/<run_id>/model` URIs may still resolve, but the *canonical* identity of a model is now `models:/<model_id>`. The Artifacts tab no longer shows model files; they live on the LoggedModel's own page. |
# MAGIC | 3 | `mlflow.evaluate(baseline_model=...)` removed; `MetricThreshold.higher_is_better` renamed to `greater_is_better` | Any evaluator-comparison code from MLflow 2 needs rewrites. We hit this directly in Module 3. |
# MAGIC
# MAGIC Other niceties of the new model: `mlflow.start_run()` is no longer required to call `log_model` (a transient run is created if needed); autolog defaults `log_traces=True` (relevant once you mix in any LLM call).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Pass the LightGBM `model_id` to Module 3
# MAGIC
# MAGIC Module 3 (tuning) needs to know which `LoggedModel` to reference as its baseline. We persist the model_id to a small "workshop state" Delta table so the handoff between notebooks is explicit and idempotent.

# COMMAND ----------

from pyspark.sql import Row

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
      key STRING NOT NULL,
      value STRING NOT NULL,
      updated_at TIMESTAMP
    )
    """
)

# Idempotent upsert via MERGE
state_upserts = [
    Row(key="lgbm_baseline_model_id", value=lgb_logged.model_id),
    Row(key="logreg_baseline_model_id", value=lr_logged.model_id),
    Row(key="lgbm_baseline_run_id", value=run_lgb.info.run_id),
]
spark.createDataFrame(state_upserts).createOrReplaceTempView("_state_upserts")
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
# MAGIC **What you just learned**
# MAGIC
# MAGIC - `LoggedModel` is a first-class MLflow 3 entity. Each `mlflow.<flavor>.log_model(..., name=...)` call produces one — with its own ID, URI, and lifecycle separate from any run.
# MAGIC - The canonical URI for loading a model is now `models:/<model_id>`. Old `runs:/<run_id>/<artifact_path>` URIs are deprecated.
# MAGIC - `mlflow.autolog()` in MLflow 3 produces LoggedModels automatically and defaults `log_traces=True`.
# MAGIC - Three breaking changes from MLflow 2.x — `name=` replacing `artifact_path=`, model storage location, and `mlflow.evaluate(baseline_model=...)` removal.
# MAGIC
# MAGIC **What's next — Module 3: Tuning & `mlflow.evaluate`**
# MAGIC
# MAGIC Module 3 tunes the LightGBM model with Optuna (15 trials), then uses `mlflow.evaluate` with a custom business metric (expected retention value) bound to the resulting `LoggedModel` via `model_id=`. Open `modules/03_tuning_and_eval/03_tuning_and_eval.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [MLflow 3 migration guide](https://mlflow.org/docs/latest/ml/mlflow-3/)
# MAGIC - [LoggedModel concept](https://mlflow.org/docs/latest/ml/model/)
# MAGIC - [Autologging in MLflow 3](https://mlflow.org/docs/latest/ml/tracking/autolog/)
