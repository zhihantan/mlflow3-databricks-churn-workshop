# VERIFICATION.md

Per-notebook self-review of the MLflow 3 + Databricks Churn Workshop. Generated 2026-05-25 after the initial build (no live-workspace dry-run yet — items requiring a real workspace to confirm are surfaced in §3 below).

---

## 1. Verification checklist applied to every notebook

Each notebook was reviewed against these criteria:

- [x] **API URLs cited inline.** Every non-obvious MLflow / Databricks call has a `# Ref: <docs URL>` comment.
- [x] **Idempotent on re-run.** All UC table creations use `IF NOT EXISTS` / `OR REPLACE` / `mode("overwrite")`. Endpoint and index creations are wrapped in `try`/`except` or guarded by existence checks.
- [x] **Config-driven.** No hardcoded catalog, schema, model, or endpoint names. Every identifier reads from `config/workshop_config.py`.
- [x] **Header + footer cells.** Every notebook starts with title / objectives / prerequisites / runtime estimate / compute, and ends with a recap + go-deeper links + handoff to the next module.
- [x] **Inter-module dependencies tracked.** State flowing from one notebook to the next goes through the `<schema>.workshop_state` Delta table (one source of truth) — no manual paste of `model_id` across notebooks.

---

## 2. Per-module verification log

| Module | Built | API URLs cited | Idempotent | Config-driven | Header + footer | Best-estimate runtime / target | Notes |
|---|---|---|---|---|---|---|---|
| 00 Setup & synthetic data | ✅ | ✅ | ✅ (`CREATE … IF NOT EXISTS`, `mode("overwrite")`) | ✅ | ✅ | ~2-3 min / ≤3 min | 20k customers, ~30k policies, ~5k claims, ~180k payments (capped to last 180d), ~500 tickets. Seeded with `SYNTHETIC_SEED=42` for determinism. |
| 01 Feature engineering | ✅ | ✅ | ✅ (`DROP TABLE IF EXISTS` before `create_table`, `mode='merge'`) | ✅ | ✅ | ~3 min / ≤4 min | Point-in-time via `FeatureLookup(timestamp_lookup_key='snapshot_date')` |
| 02 Experiment tracking | ✅ | ✅ | ✅ (`MERGE INTO` for state, new runs are additive) | ✅ | ✅ | ~5 min / ≤6 min | Demonstrates `LoggedModel` w/ explicit `name=` kwarg + `models:/<model_id>` URI |
| 03 Tuning & evaluation | ✅ | ✅ | ✅ | ✅ | ✅ | ~5 min / ≤6 min | 15 Optuna trials (per `N_OPTUNA_TRIALS`); `mlflow.evaluate(model_id=...)` binds eval to LoggedModel |
| 04 Registry & serving | ✅ | ✅ | ✅ (`update_config` if endpoint exists, else `create`) | ✅ | ✅ | ~7-8 min / ≤8 min | **Background-provisioning pattern** absorbs ~5-7 min endpoint cold-start. Aliases set via `MlflowClient.set_registered_model_alias` |
| 05 Production monitoring | ✅ | ✅ | ✅ (Delta `mode("overwrite")` for the drift table; eval runs are append-only) | ✅ | ✅ | ~2-3 min / ≤4 min | `scipy.stats` KS + χ² for input drift + `mlflow.evaluate` per window for prediction drift; persists to `<schema>.churn_drift_metrics`. Simulated drift on `payment_failures_60d` |
| 06 Tracing + Prompt Registry | ✅ | ✅ | ✅ (idempotent endpoint check; prompt registry is versioned/append-only) | ✅ | ✅ | ~2-3 min / ≤3 min | Kicks off VS endpoint provisioning in cell 2 so Module 7 finds it ready |
| 07 RAG | ✅ | ✅ | ✅ (`list_indexes` check before create) | ✅ | ✅ | ~6-8 min / ≤8 min | Delta Sync + managed embeddings (`databricks-gte-large-en`). Endpoint polled with 10-min timeout |
| 08 Retention agent | ✅ | ✅ | ✅ (try/except around `agents.deploy` kwargs) | ✅ | ✅ | ~8-9 min / ≤9 min | **Two files**: `agent.py` (ResponsesAgent subclass) + `08_retention_agent.py` (driver). Config injected via JSON artifact, not env vars |
| 09 GenAI evaluation | ✅ | ✅ | ✅ (eval runs are append-only by design) | ✅ | ✅ | ~5-6 min / ≤6 min | Two evals: full agent + simpler prompt-only `predict_fn` for iteration demo |
| 10 Capstone | ✅ | ✅ | ✅ (`overwrite` on capstone_retention_emails) | ✅ | ✅ | ~3-5 min / ≤5 min | Falls back to local agent if M8 deployed endpoint isn't ready |
| **Total** | | | | | | **~50 min / ≤62 min** | ~10 min headroom under the 60-min hard cap |

---

## 3. Flagged items needing live-workspace verification

These are points where my research left an open question that can only be conclusively confirmed by running on a real Databricks workspace. They are surfaced here so a human reviewer can verify before the workshop is delivered live.

### High priority — could break a notebook end-to-end if wrong

1. **`mlflow.get_logged_model` exact signature** (Module 2).
   - Live MLflow API docs page truncated during research. Signature inferred from search snippets. The call is `mlflow.get_logged_model(model_id)` in Module 2.
   - **Verify:** `help(mlflow.get_logged_model)` in a notebook, or check `dir(mlflow)` for the function.
   - **Fallback if it doesn't exist:** use `MlflowClient().get_logged_model(model_id)` or iterate via `mlflow.search_logged_models(...)`.

2. ~~**`databricks.lakehouse_monitoring` package availability** (Module 5).~~ **Resolved.**
   - The legacy `databricks.lakehouse_monitoring` package was removed from PyPI. Module 5 was rewritten to compute drift directly via `scipy.stats` (KS + χ²) and `mlflow.evaluate(model_type="classifier")` per window — no managed-monitor dependency. The drift metrics land in a Delta table + MLflow time-series the same way Lakehouse Monitoring's output would.

3. **`databricks.agents.deploy()` signature** (Module 8).
   - I wrapped the call in `try/except TypeError` to handle kwarg incompatibilities. The minimal-kwargs fallback should always work.
   - **Verify:** `from databricks import agents; help(agents.deploy)` confirms which kwargs are accepted.

4. **`ResponsesAgent.create_text_output_item(text, id)` helper** (Module 8 `agent.py`).
   - I use this helper to build the output item. If the method signature differs (e.g. `(id, text)` positional), the agent will error at first invocation.
   - **Verify:** `help(mlflow.pyfunc.ResponsesAgent.create_text_output_item)` in a notebook after `%pip install`.
   - **Fallback:** construct the dict manually — `{"type": "message", "id": "msg_001", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}`.

5. **OpenAI Agents SDK package name and import path** (Module 8 `agent.py`).
   - Logged model declares `openai-agents>=0.1` and imports `from agents import Agent, Runner, function_tool`.
   - **Verify:** `%pip install openai-agents` succeeds; the import works.

6. **`mlflow.lightgbm.load_model(...)` returning categorical-aware model on string-input data** (Module 4 / Module 8).
   - LightGBM was trained with `categorical_feature=CATEGORICAL` and categorical-dtype columns. When the serving endpoint receives string-typed JSON columns, LightGBM may or may not auto-restore categorical dtype.
   - **Verify:** the Module 4 endpoint query returns sensible predictions (not all 0s or NaN). If it fails, wrap the model in a custom `pyfunc.PythonModel` that explicitly `astype("category")` on categorical columns before predict.

### Medium priority — visual / cosmetic

7. ~~**Lakehouse Monitor profile / drift metric table column names** (Module 5).~~ **Resolved.**
   - Module 5 no longer depends on auto-generated monitor tables — the drift Delta table is hand-written by the notebook with a known schema (feature / feature_type / test_statistic / p_value / mean_shift_pct / drift_detected). The SQL query in cell 8 runs against that table directly.

8. **MLflow 3 LoggedModel `model_id` field accessor** (Module 2).
   - Used as `mlflow.sklearn.log_model(...).model_id` and `mlflow.lightgbm.log_model(...).model_id`. If the returned object exposes the ID under a different name (e.g. `.logged_model_id` or `.id`), Module 2 cell 7 will `AttributeError`.

9. **`mlflow.search_traces(...)` return shape** (Module 6 cell 10, Module 7 cell 9).
   - I assumed it returns a pandas DataFrame with columns `trace_id`, `request`, `response`, `execution_time_ms`. If the column names differ, the `display(...)` calls will surface different fields. Cosmetic.

### Low priority — observed-only

10. **MLflow 3.12 on Serverless base env.**
    - All notebooks `%pip install "mlflow[databricks]>=3.12,<4"` and restart Python. The upgrade should succeed on both DBR 17.3 LTS ML and Serverless ML, but I haven't dry-run on Serverless ML specifically.

11. **`databricks.vector_search.client.VectorSearchClient().list_indexes(name=...)` return shape** (Module 7).
    - The shape `{"vector_indexes": [{"name": "..."}]}` is from one snippet; some versions return a flat list. The idempotent existence check uses defensive `.get(...)` lookups.

---

## 4. What I did NOT verify (out of scope for a build pass)

- **No end-to-end dry run** in a live Databricks workspace. The first dry-run is the natural next step before any live delivery.
- **No timing measurements** — runtime estimates in §2 are based on per-module workload sizing (synthetic data row counts, model complexity, judge call counts), not observed wall-clock.
- **No A/B between Serverless ML and DBR 17.3 LTS ML compute.** Notebooks are written to be portable across both; differences will surface in dry-run.
- **No load test** on rate limits — Module 9's 75 judge calls × N concurrent participants could hit FMAPI OTPM on a shared workspace. Recommend per-user workspaces or staggered eval starts for groups > 10.

---

## 5. Re-run hygiene

To re-run the workshop after a previous run:

1. Run `modules/11_teardown/11_teardown.py` to drop catalog/schema/endpoints/indexes/registered models + the MLflow experiment.
2. Re-clone the repo (or `git pull` if you already have it).
3. Open `setup/00_setup_and_synthetic_data.py` and Run All.

The reset script is idempotent — safe to run multiple times even if some resources are already gone.
