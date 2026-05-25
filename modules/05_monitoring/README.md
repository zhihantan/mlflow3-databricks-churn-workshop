# Module 05 — Production Monitoring with Lakehouse Monitoring

Simulate a 2-window inference table where the second window has a deliberately shifted `payment_failures_60d` distribution, create a Lakehouse Monitor with `InferenceLogConfig`, and query the auto-generated drift metric table to see the shift.

This module uses a **simulated** inference table (locked Q6) rather than wiring real endpoint traffic from Module 4 — keeps the module under budget and the drift signal deterministic for didactic clarity.

**Concepts covered**
- Lakehouse Monitoring profile types: `TimeSeries`, `Snapshot`, **`InferenceLog`**
- `WorkspaceClient.data_quality.create_monitor` with `DataProfilingConfig(inference_log=InferenceLogConfig(...))`
- `InferenceProblemType.INFERENCE_PROBLEM_TYPE_CLASSIFICATION`
- Output tables produced by a monitor: profile metrics + drift metrics
- Querying drift metrics for a specific feature
- Discussion: production monitoring patterns (alert thresholds, slicing, retraining triggers)

**Prerequisites**
- Modules 0, 1, 4 have been run (we reuse the registered model name to tag inference rows with a `model_id`).

**Runtime target**: ~4 minutes.
**Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

**Notebook**: [`05_monitoring.py`](./05_monitoring.py)

---

> Status: scaffold stub.
