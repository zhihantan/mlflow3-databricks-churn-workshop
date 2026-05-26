# Module 05 — Production Monitoring with MLflow + scipy

Simulate a 2-window inference table where the second window has a deliberately shifted `payment_failures_60d` distribution, compute per-feature drift metrics with `scipy.stats`, persist them to a queryable Delta table, and run `mlflow.evaluate(model_type="classifier")` per window to detect *prediction-level* drift on top of input drift.

This module uses a **simulated** inference table (locked Q6) rather than wiring real endpoint traffic from Module 4 — keeps the module under budget and the drift signal deterministic for didactic clarity.

**Concepts covered**
- Delta inference table with Change Data Feed enabled — the durable, queryable home for every model prediction.
- Two complementary drift signals:
  - **Input drift** — `scipy.stats.ks_2samp` (Kolmogorov–Smirnov) for numerics, `scipy.stats.chi2_contingency` (chi-squared) for categoricals.
  - **Prediction drift / performance shift** — `mlflow.evaluate(model_type="classifier")` per window for accuracy / F1 / log-loss / AUC deltas.
- Persisting drift to a Delta `<schema>.churn_drift_metrics` table + logging it to MLflow as a time series.
- Window-over-window deltas computed inline so the regression is visible at a glance.
- Production retraining-trigger pattern: Databricks SQL Alert on the drift table fires when `features_with_drift > 0`.

**Prerequisites**
- Modules 0, 1, 4 have been run (we reuse the registered model name to tag inference rows with a `model_id`).

**Runtime target**: ~2-3 minutes (everything runs synchronously; no async monitor refresh to wait for).
**Compute**: Serverless or DBR 17.3 LTS ML. `scipy` is preinstalled on both.

**Notebook**: [`05_monitoring.py`](./05_monitoring.py)
