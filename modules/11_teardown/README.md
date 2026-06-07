# Module 11 — Teardown (destroy all workshop resources)

Idempotent, dependency-safe teardown of **everything** the workshop creates for the current user. Run it on demand to reclaim resources or to get a clean slate for a fresh re-run. A confirmation widget guards against accidental runs.

**Run separately — NOT part of the e2e job.** Destroying resources is deliberately a manual, opt-in action, so this module is excluded from both job definitions (`resources/workshop_e2e_job.yml` and `scripts/deploy_workshop_job.py`). Open the notebook, set the `confirm_schema` widget to your `FULL_SCHEMA`, and `Run All`.

**What it removes (in order)**
1. Serving endpoints — agent (Module 8) + churn (Module 4)
2. Vector Search — index then endpoint (Modules 6/7)
3. Registered models in UC — `bolttech_retention_agent`, `bolttech_churn_model`
4. Prompt Registry entries — summary / RAG / email (Modules 6, 7)
5. Scheduled scorers — stop + delete production monitors (Module 9 §9)
6. Lakehouse monitor — on the inference table (Module 5)
7. **MLflow experiment** — deleted so its path is freed for a clean UC-trace re-bind
8. UC schema (cascade) — every table/view/feature-table, incl. the `mlflow_traces_*` UC trace tables and `workshop_state`

**Why it also deletes the MLflow experiment.** A UC trace destination can only bind to a *trace-free* experiment, so an existing experiment (with traces) is exactly what silently forces Module 0 back onto the default trace store. Deleting it here lets the next Module 0 run bind UC trace storage cleanly. This is the single, canonical teardown for the workshop.

**Idempotent** — every step skips resources that don't exist; re-running is safe.

**Does NOT remove** the `MLFlow Workshop e2e job` (Workflows) or the Git folder — those are reusable infrastructure, not per-run data.

**Prerequisites**
- None. Safe to run at any point; skips whatever isn't there.

**Runtime target**: ~2-4 minutes (the agent serving endpoint delete is the slowest step).
**Compute**: Serverless or DBR 17.3 LTS ML.

**Notebook**: [`11_teardown.py`](./11_teardown.py)
