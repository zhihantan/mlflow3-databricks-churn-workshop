# Module 02 — Experiment Tracking & the MLflow 3 `LoggedModel`

Train a baseline logistic regression and a LightGBM classifier on the Module 1 training set, demonstrating the MLflow 3 `LoggedModel` entity and the new `models:/<model_id>` URI scheme.

This is the **flagship MLflow 3 pedagogy module** — it's where the difference between MLflow 2's run-scoped model artifacts and MLflow 3's first-class `LoggedModel` becomes visible. A side-by-side markdown block makes the contrast explicit.

**Concepts covered**
- The MLflow 3 `LoggedModel` entity — first-class lifecycle, independent of any single run
- `mlflow.autolog()` in MLflow 3 (`log_traces=True` default; new `exclude_flavors`)
- `mlflow.sklearn.log_model(..., name=...)` and `mlflow.lightgbm.log_model(..., name=...)` — `name=` replaces `artifact_path=`
- Loading via `models:/<model_id>` instead of `runs:/<run_id>/model`
- Run comparison + LoggedModel retrieval (`mlflow.get_logged_model`)
- The three biggest MLflow 2 → 3 breaking changes participants need to know

**Prerequisites**
- Modules 0, 1 have been run.

**Runtime target**: ~6 minutes.
**Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

**Notebook**: [`02_experiment_tracking.py`](./02_experiment_tracking.py)

---

> Status: scaffold stub.
