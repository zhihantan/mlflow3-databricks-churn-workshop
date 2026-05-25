# Databricks notebook source
# MAGIC %md
# MAGIC # Module 03 — Hyperparameter Tuning + `mlflow.evaluate`
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Run an Optuna study with **15 trials**, each logged as a nested MLflow run.
# MAGIC - Train a tuned LightGBM model and log it as a new `LoggedModel`.
# MAGIC - Use **`mlflow.evaluate`** to compute classification metrics + a **custom business metric** (expected retention value), and bind the eval results to the LoggedModel via the new `model_id=` parameter.
# MAGIC - See first-hand the MLflow 3 breaking change `baseline_model=` was removed from `mlflow.evaluate` — and what to do instead.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 0, 1, 2 have been run.
# MAGIC
# MAGIC **Expected runtime**: ~5 minutes (15 Optuna trials × ~10s + eval).
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "lightgbm>=4.6" \
# MAGIC   "optuna>=3.6" \
# MAGIC   "scikit-learn>=1.6"
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
    EXPERIMENT_PATH,
    N_OPTUNA_TRIALS,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load training data + read baseline model_id from workshop state

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

mlflow.set_experiment(EXPERIMENT_PATH)

TRAINING_TABLE = f"{FULL_SCHEMA}.churn_training_set"
STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"

df = spark.table(TRAINING_TABLE).toPandas()

CATEGORICAL = ["country", "plan_tier", "primary_device"]
NUMERIC = [
    "age", "policy_tenure_days", "active_policy_count", "total_policy_count", "avg_premium",
    "claims_count_90d", "claim_amount_sum_90d", "pending_claims_90d",
    "payment_failures_60d", "payments_count_60d",
    "support_ticket_count_30d", "negative_ticket_share_30d",
]
TARGET = "churned"

X = df[CATEGORICAL + NUMERIC].copy()
y = df[TARGET].astype(int)
# Normalize categoricals to object/string dtype to match the LR/LGBM pipelines from Module 2.
for col in CATEGORICAL:
    X[col] = X[col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Churn rate: {y_train.mean():.1%}")

# Recover the Module 2 baseline model_id so we can compare tuned vs baseline at the end.
baseline_model_id = (
    spark.table(STATE_TABLE)
    .filter("key = 'lgbm_baseline_model_id'")
    .select("value")
    .first()[0]
)
print(f"Baseline LGBM model_id (from Module 2): {baseline_model_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define the Optuna objective with nested MLflow runs
# MAGIC
# MAGIC Each Optuna trial is one `mlflow.start_run(nested=True)`. We manually `log_params` and `log_metric` — Optuna doesn't yet have a first-party MLflow autolog hook in DBR 17.3 LTS ML's environment, so the manual logging here is also pedagogically useful (shows what autolog does under the hood).

# COMMAND ----------

import lightgbm as lgb
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Suppress Optuna's per-trial logging — MLflow runs are the source of truth.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_lgb_pipeline(params: dict) -> Pipeline:
    """OneHotEncoder + LightGBM Pipeline — mirrors Module 2's LGBM pipeline shape
    so signatures, predict, and serving stay uniform across modules."""
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ])),
        ("clf", lgb.LGBMClassifier(**params, objective="binary", random_state=42, verbosity=-1)),
    ])


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
    }
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        pipe = _make_lgb_pipeline(params)
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_auc", auc)
    return auc


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run the Optuna study

# COMMAND ----------

with mlflow.start_run(run_name="lgbm_tuning_study") as parent_run:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_params = study.best_params
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_test_auc", study.best_value)

print(f"\nBest test AUC across {N_OPTUNA_TRIALS} trials: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train the final tuned model → new `LoggedModel`
# MAGIC
# MAGIC We retrain on the best Optuna params and log as a fresh `LoggedModel`. Note again `name="lgbm_tuned"` — MLflow 3's replacement for `artifact_path=`.

# COMMAND ----------

from mlflow.models import infer_signature

with mlflow.start_run(run_name="lgbm_tuned") as run_tuned:
    mlflow.log_params(best_params)
    final_pipe = _make_lgb_pipeline(best_params)
    final_pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:, 1])
    mlflow.log_metric("test_auc", auc)

    # Explicit signature — required for Unity Catalog registration in Module 4.
    sig_input = X_train.head(3)
    tuned_signature = infer_signature(sig_input, final_pipe.predict(sig_input))

    # Pipeline is a sklearn estimator → use the sklearn flavor for logging.
    # Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
    #
    # `extra_pip_requirements=["lightgbm"]` is required: the sklearn flavor's
    # dependency inference can't see transitive deps inside a Pipeline step, so it
    # would otherwise omit lightgbm from requirements.txt — causing Model Serving
    # to fail with ModuleNotFoundError at container load time.
    tuned_logged = mlflow.sklearn.log_model(
        sk_model=final_pipe,
        name="lgbm_tuned",
        input_example=sig_input,
        signature=tuned_signature,
        extra_pip_requirements=[f"lightgbm=={lgb.__version__}"],
    )

print(f"Tuned model_id: {tuned_logged.model_id}")
print(f"Tuned model_uri: {tuned_logged.model_uri}")
print(f"Tuned test AUC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Define a custom business metric: *expected retention value*
# MAGIC
# MAGIC AUC is a fine ML metric but doesn't directly answer "what's this model worth in dollars?". We add a custom metric using `mlflow.metrics.make_metric`:
# MAGIC
# MAGIC > **Expected retention value** = (true positives × avg LTV × intervention success rate) − (false positives × outreach cost) − (all flagged × outreach cost)
# MAGIC
# MAGIC Assumptions (configurable; what matters here is the **pattern** of wiring a business metric into `mlflow.evaluate`):
# MAGIC - Average customer LTV = $200
# MAGIC - Intervention success rate = 25% of contacted churners retain
# MAGIC - Outreach cost = $5 per contacted customer
# MAGIC
# MAGIC Higher is better.
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html

# COMMAND ----------

from mlflow.metrics import make_metric

AVG_LTV = 200.0
SUCCESS_RATE = 0.25
OUTREACH_COST = 5.0


def expected_retention_value_fn(predictions, targets, metrics=None):
    """Expected dollar value of running a retention campaign on the model's predicted churners.

    Parameters
    ----------
    predictions : pandas.Series
        Class predictions from the model (0/1).
    targets : pandas.Series
        Ground-truth labels (0/1).
    metrics : dict | None
        Already-computed builtin metrics. Unused here; required by the make_metric signature.
    """
    preds = predictions.values if hasattr(predictions, "values") else predictions
    truth = targets.values if hasattr(targets, "values") else targets
    flagged = (preds == 1)
    tp = int(((preds == 1) & (truth == 1)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    total_flagged = int(flagged.sum())
    return float(tp * AVG_LTV * SUCCESS_RATE - fp * OUTREACH_COST - total_flagged * OUTREACH_COST / 2)


expected_retention_value = make_metric(
    eval_fn=expected_retention_value_fn,
    greater_is_better=True,
    name="expected_retention_value",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run `mlflow.evaluate` and bind results to the tuned `LoggedModel`
# MAGIC
# MAGIC The MLflow 3 `model_id=` parameter is the key here — it attaches the evaluation metrics, tables, and dataset reference to the LoggedModel rather than just to a run. That gives us per-model lineage that survives across runs.
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html

# COMMAND ----------

import mlflow.models

# With the Pipeline approach, X_test has plain object/numeric dtypes — no Categorical
# columns to confuse the evaluator. We can pass `model=` directly and let MLflow
# auto-compute the full classification metric suite (ROC-AUC, log-loss, etc.) via
# predict_proba.
eval_data = X_test.copy()
eval_data["churned"] = y_test.values

with mlflow.start_run(run_name="lgbm_tuned_evaluate"):
    eval_results = mlflow.models.evaluate(
        model=f"models:/{tuned_logged.model_id}",
        data=eval_data,
        model_type="classifier",
        targets="churned",
        extra_metrics=[expected_retention_value],
        model_id=tuned_logged.model_id,   # NEW in MLflow 3 — binds eval to LoggedModel
    )

print("\nBuiltin + custom metrics:")
for name, value in sorted(eval_results.metrics.items()):
    print(f"  {name:40s} = {value:.4f}" if isinstance(value, (int, float)) else f"  {name:40s} = {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Quick "tuned vs baseline" comparison
# MAGIC
# MAGIC `mlflow.evaluate(baseline_model=...)` was **removed in MLflow 3** — we compute the comparison ourselves now. Pass each model through `mlflow.pyfunc.load_model` via its `model_id` URI.

# COMMAND ----------

# Both models are now sklearn Pipelines logged via mlflow.sklearn — load them via the
# sklearn flavor to get back the full Pipeline object (so we can call .predict_proba()
# for ROC-AUC and not just .predict() for class labels).
baseline_pipe = mlflow.sklearn.load_model(f"models:/{baseline_model_id}")
tuned_pipe = mlflow.sklearn.load_model(f"models:/{tuned_logged.model_id}")

baseline_preds = baseline_pipe.predict(X_test)
tuned_preds = tuned_pipe.predict(X_test)
baseline_proba = baseline_pipe.predict_proba(X_test)[:, 1]
tuned_proba = tuned_pipe.predict_proba(X_test)[:, 1]

baseline_auc = roc_auc_score(y_test, baseline_proba)
tuned_auc = roc_auc_score(y_test, tuned_proba)

baseline_erv = expected_retention_value_fn(pd.Series(baseline_preds), y_test)
tuned_erv = expected_retention_value_fn(pd.Series(tuned_preds), y_test)

import pandas as pd
comparison_df = pd.DataFrame(
    {
        "model": ["lgbm_baseline (Module 2)", "lgbm_tuned (Module 3)"],
        "model_id": [baseline_model_id, tuned_logged.model_id],
        "test_auc": [baseline_auc, tuned_auc],
        "expected_retention_value": [baseline_erv, tuned_erv],
    }
)
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Persist the tuned `model_id` for Module 4

# COMMAND ----------

from pyspark.sql import Row

upserts = [
    Row(key="lgbm_tuned_model_id", value=tuned_logged.model_id),
    Row(key="lgbm_tuned_run_id", value=run_tuned.info.run_id),
    Row(key="lgbm_tuned_test_auc", value=str(tuned_auc)),
]
spark.createDataFrame(upserts).createOrReplaceTempView("_state_upserts")
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
# MAGIC - Optuna + MLflow integrate cleanly via nested `mlflow.start_run(nested=True)`. Every trial gets its own run; the parent run carries the best params.
# MAGIC - `mlflow.evaluate(model_id=...)` in MLflow 3 binds evaluation metrics to a specific `LoggedModel` for end-to-end lineage.
# MAGIC - Custom business metrics via `mlflow.metrics.make_metric(eval_fn=..., greater_is_better=...)` plug straight into `extra_metrics=[...]` on `mlflow.evaluate`.
# MAGIC - `baseline_model=` was removed from `mlflow.evaluate` in MLflow 3. Side-by-side comparisons are now done by loading both LoggedModels by `model_id` and computing the comparison yourself.
# MAGIC
# MAGIC **What's next — Module 4: UC Registry + Model Serving**
# MAGIC
# MAGIC Module 4 registers the tuned model in Unity Catalog, sets `@champion` / `@challenger` aliases, and provisions a Model Serving endpoint. Open `modules/04_registry_and_serving/04_registry_and_serving.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [`mlflow.evaluate` reference](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html)
# MAGIC - [Custom metrics](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html)
# MAGIC - [Optuna + MLflow](https://optuna.readthedocs.io/en/stable/reference/integration.html)
