# Module 09 — GenAI Evaluation with `mlflow.genai.evaluate`

Run `mlflow.genai.evaluate` against the locally-loaded Module 8 agent on a hand-curated 25-example eval set, using a mix of built-in and custom scorers. Iterate on the registered prompt and re-evaluate to demonstrate the eval-driven prompt-iteration loop.

This module uses the **locally-loaded** agent (not the deployed endpoint) so the evaluation runs fast and is independent of Module 8's deploy timing.

**Concepts covered**
- `mlflow.genai.evaluate(data, scorers, predict_fn, model_id)` — the new unified eval API
- Eval dataset schema: `inputs` / `expectations` / `outputs` columns
- `predict_fn` contract: receives `inputs` dict unpacked as kwargs
- Built-in scorers:
  - `Correctness()` — uses `expectations.expected_facts`
  - `Safety()` — Databricks-managed
  - `RetrievalGroundedness()` — RAG fidelity
  - `RelevanceToQuery()` — response relevance
- Custom `Guidelines(name="bolttech_voice", guidelines="...")` scorer for bolttech-specific tone rules
- Prompt iteration: bump the registered prompt to v2 → re-run eval → compare runs in the MLflow UI
- **(Optional §9) Production monitoring** — `scorer.register(name=...).start(sampling_config=ScorerSamplingConfig(...))` schedules the same scorers to run continuously on a sample of incoming traces, feeding the experiment's **Overview → Quality** dashboard. Async (~15-20 min) + adds judge cost — pre-stage before a live session; teardown is in `scripts/reset_workshop.py`.
  - **Requires Unity Catalog trace storage**, which is set up in **Module 0**. `config/workshop_config.py` auto-resolves a usable SQL warehouse, so this is on by default (pin one with the `MONITORING_WAREHOUSE_ID` env var, or disable with `WORKSHOP_DISABLE_MONITORING=1`). When no warehouse is usable, scorer assessments still attach per-trace (Traces tab), but the aggregate Overview charts (Usage / Quality) stay empty. See [Store MLflow traces in Unity Catalog](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog).

**Files in this folder**
- `09_genai_evaluation.py` — driver notebook
- `eval_dataset.py` — eval-set **builder** (`build_examples(customer_ids)`) with the query patterns, `expected_facts`, and bolttech-voice guidelines. Customer IDs are **not hardcoded**: Module 9 derives them at runtime from customers the agent can serve (those with support tickets), so `Correctness` measures response quality rather than a data mismatch

**Prerequisites**
- Modules 6, 7, 8 have been run.

**Runtime target**: ~6 minutes (25 examples × ~4 scorers ≈ 100 judge calls; warn about FMAPI OTPM on shared workspaces).
**Compute**: Serverless or DBR 17.3 LTS ML.

**Notebook**: [`09_genai_evaluation.py`](./09_genai_evaluation.py)

---

> Status: scaffold stub.
