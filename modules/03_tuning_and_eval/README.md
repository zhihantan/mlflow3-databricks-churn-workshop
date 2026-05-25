# Module 03 — Hyperparameter Tuning + `mlflow.evaluate`

Tune the Module 2 LightGBM classifier with Optuna (15 trials), then evaluate the resulting `LoggedModel` with `mlflow.evaluate` using a custom business metric (*expected retention value*) bound to the LoggedModel via the new `model_id=` parameter.

**Concepts covered**
- Optuna + MLflow integration via nested `mlflow.start_run(nested=True)` per trial
- `mlflow.evaluate(model_type="classifier", model_id=...)` — binding evaluation results to a LoggedModel for lineage
- Built-in classification metrics (precision, recall, F1, ROC-AUC, log-loss, etc.)
- Custom metric API: `mlflow.metrics.make_metric(eval_fn=..., greater_is_better=True, ...)`
- Why `baseline_model` was removed in MLflow 3 (call out the breaking change)

**Prerequisites**
- Modules 0, 1, 2 have been run.

**Runtime target**: ~6 minutes (15 Optuna trials × ~10s each + final eval).
**Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

**Notebook**: [`03_tuning_and_eval.py`](./03_tuning_and_eval.py)

---

> Status: scaffold stub.
