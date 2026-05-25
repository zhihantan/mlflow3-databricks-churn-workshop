# PLAN.md — MLflow 3 + Databricks Churn Workshop for bolttech

> Status: **Plan only — no scaffolding yet.** Please review, push back where you disagree, and answer the open questions in §8. Once approved I will scaffold and build module-by-module.

| | |
|---|---|
| Repo | `mlflow3-databricks-churn-workshop` |
| License | Apache 2.0 |
| Audience | bolttech ML/data engineers — intermediate Python/ML, new to MLflow 3 + GenAI on Databricks |
| Use case | Policyholder churn prediction → GenAI retention outreach |
| Compute target | Serverless notebook compute (ML base env) with classic DBR 17.3 LTS ML fallback |
| Participant runtime budget | ≤45 min code execution (60 min hard cap) |
| Plan author tool budget | Unlimited — slow careful build → smooth fast workshop |

---

## 1. Executive Summary

A ten-module workshop teaching MLflow 3 on Databricks end-to-end against a single bolttech-style churn narrative. Part A (Modules 0–5) covers classic ML with the new `LoggedModel` entity, Feature Engineering in UC, UC Model Registry, Model Serving, and Lakehouse Monitoring. Part B (Modules 6–10) covers GenAI with MLflow Tracing, Prompt Registry, RAG over synthetic support tickets, a `ResponsesAgent` retention-outreach agent that calls back into the Module 4 endpoint and Module 7 vector index, and `mlflow.genai.evaluate` with LLM-as-judge scorers. Capstone stitches it all into a single batch-score-and-draft-emails workflow.

The repo is cloned into a Databricks workspace via Git folders and every notebook runs top-to-bottom on a fresh workspace with no edits. A single `config/workshop_config.py` is the single source of truth for catalog, schema, model names, and endpoint names.

---

## 2. Technical Decisions

### 2.1 Locked (will commit unless you object)

| Decision | Choice | One-line rationale |
|---|---|---|
| MLflow version | **3.12.0** (pinned floor `mlflow[databricks]>=3.12`) | DBR 17.3 LTS ML ships 3.0.1 only; need 3.12 for current `mlflow.genai` surface and predefined scorers |
| Runtime | **DBR 17.3 LTS ML** (classic fallback) + Serverless ML (Beta) base env | Current LTS; ships sklearn 1.6.1 / LightGBM 4.6.0 / XGBoost 3.0.0 / Optuna 3.6.1 — see [release notes](https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml) |
| Chat model (FMAPI) | **`databricks-claude-haiku-4-5`** (primary) with `databricks-meta-llama-3-3-70b-instruct` as fallback | GA pay-per-token, fast + cheap → keeps GenAI modules under budget; Llama 3.3 70B is a no-preview-risk fallback |
| Embedding model | **`databricks-gte-large-en`** | GA, paired with managed Delta Sync indexes in current docs |
| Vector Search index | **Delta Sync, `TRIGGERED`, managed embeddings** | Cheapest provisioning, sub-minute sync on ~500 rows |
| LLM client pattern | **`openai.OpenAI(base_url=...)` via the `databricks-openai` helper** | Idiomatic 2026 pattern per Databricks docs; `mlflow.deployments` still works but no longer recommended |
| Agent flavor | **`mlflow.pyfunc.ResponsesAgent`** | New 2026 canonical pattern (replaced `ChatAgent`); see [author-agent docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) |
| Tuning library | **Optuna** with manual MLflow logging per trial | Optuna ships in DBR 17.3 ML; autolog not yet first-party for Optuna so we'll demo the manual pattern (also more transparent for participants) |
| Notebook format | **Databricks notebook source** (`.py` with `# Databricks notebook source` + `# COMMAND ----------`) | Renders as notebooks via Git folders; still the recommended pattern |
| Synthetic data scale | **20k customers, ~30k policies, ~60k payment rows, ~500 support tickets** | Enough to look real and train meaningfully; small enough to fit module budgets |
| Eval set size | **~25 hand-curated examples** for `mlflow.genai.evaluate` | Per the prompt's guidance |
| Optuna trials | **15 trials** | Per the prompt's "10–20" guidance; produces a visible-but-fast improvement |

### 2.2 Resolved forks (you locked these on 2026-05-24/25)

1. **Inner-agent framework** — **OpenAI Agents SDK** (`openai-agents`).
2. **Catalog/schema strategy** — **Per-user schemas** `bolttech_workshop.churn_<sanitized_user>`.
3. **Agent endpoint deployment** — **Actually deploy via `agents.deploy()`**. Module 8 will absorb the cold-start by kicking off deploy in cell 1, doing all other M8 work while it provisions, and querying the live endpoint at the end. Module 10 uses the deployed endpoint as the production demo. Module 8 runtime budget raised from ≤5 min → **≤9 min**.
4. **Module 5 monitoring approach** — **Simulated inference table with synthetic drift**.
5. **Catalog name** — committing to default `bolttech_workshop` (no pushback received).
6. **`customer_id` format** — committing to default zero-padded strings `CUST_000001` (no pushback received).

---

## 3. Curriculum Outline

Each module = one Databricks-notebook-source `.py` file inside `modules/<NN_name>/`, plus a short `README.md` with learning objectives, prerequisites, expected runtime, and key MLflow 3 / Databricks concepts. Each notebook starts with a YAML-style header cell (title, objectives, prerequisites, expected runtime, required compute) and ends with a "what you just learned" recap + "go deeper" links + handoff to the next module.

### Part A — Classic ML with MLflow 3

#### Module 0 — Setup & Synthetic Data Generation
- **File:** `setup/00_setup_and_synthetic_data.py`
- **Runtime target:** ≤3 min
- **Concepts:** UC catalog/schema bootstrapping, idempotent setup, seeded synthetic data
- **APIs used:** `spark.sql("CREATE CATALOG/SCHEMA IF NOT EXISTS ...")`, plain Spark/pandas writes to Delta
- **Outputs:** Five Delta tables in `<catalog>.<schema>`: `customers`, `policies`, `claims`, `payments`, `support_tickets`, plus a snapshot table `customer_snapshots` (the "as-of-date" training base)
- **Synthetic-data design:**
  - 20k customers across SG, MY, ID, IN, JP, TH, AU, US, UK with realistic age/tenure distributions
  - Devices, mobile plans, and embedded-insurance products mirroring bolttech's actual lines of business (no real customer data)
  - Realistic churn drivers: rising payment failures, recent unresolved claim, support tickets with negative sentiment
  - Reproducible — fixed `numpy.random` seed
- **Idempotency:** Each table written with `mode("overwrite")` + Delta `OR REPLACE`. Re-running is safe.

#### Module 1 — Feature Engineering in Unity Catalog
- **Folder:** `modules/01_feature_engineering/`
- **Runtime target:** ≤4 min
- **Concepts:** UC feature tables, primary keys + timeseries columns, point-in-time lookups
- **APIs used:**
  - `from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup`
  - `fe.create_table(name=..., primary_keys=['customer_id', 'snapshot_date'], timeseries_columns='snapshot_date', schema=df.schema)` — Ref: [Feature tables in UC](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc)
  - `fe.write_table(name=..., df=..., mode='merge')`
  - `fe.create_training_set(df=labels, feature_lookups=[FeatureLookup(..., timestamp_lookup_key='snapshot_date')], label='churned')` — Ref: [Time-series feature lookups](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series)
- **Features produced (per-customer, per-snapshot-date):** `policy_tenure_days`, `active_policy_count`, `claims_count_90d`, `claim_amount_sum_90d`, `payment_failures_60d`, `avg_premium`, `support_ticket_count_30d`, `negative_ticket_share_30d`, `country`, `plan_tier`
- **Outputs:** UC feature table `<cat>.<sch>.customer_churn_features` + a saved `training_df` view for Module 2

#### Module 2 — Experiment Tracking & LoggedModel
- **Folder:** `modules/02_experiment_tracking/`
- **Runtime target:** ≤6 min
- **Concepts:** **The MLflow 3 `LoggedModel` entity** (THE flagship pedagogy of the workshop), autologging, multi-model run comparison, `models:/<model_id>` URI
- **APIs used:**
  - `mlflow.set_experiment(f"/Users/{user}/mlflow3_workshop")`
  - `mlflow.autolog()` — Ref: [Autolog in MLflow 3](https://mlflow.org/docs/latest/ml/tracking/autolog/) — note `log_traces=True` default
  - `mlflow.sklearn.log_model(lr_pipe, name="logreg_baseline", input_example=X_train.head())` — Ref: [`mlflow.sklearn`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html). **`name=` replaces `artifact_path=`** — workshop must call this out explicitly
  - `mlflow.lightgbm.log_model(lgbm, name="lgbm_baseline", ...)`
  - `mlflow.get_logged_model(model_id)` — retrieve the new entity (FLAG: re-verify exact signature; my research found search-snippet evidence but the live API page rendered partially)
  - `mlflow.set_active_model(...)` — Ref: [tracking docs](https://mlflow.org/docs/latest/ml/tracking/)
  - Load via `mlflow.pyfunc.load_model(f"models:/{model_id}")` — the new URI scheme
- **Pedagogy block (in a markdown cell):** side-by-side "what would this have looked like in MLflow 2" vs 3 — `runs:/<run_id>/model` (gone) vs `models:/<model_id>` (new), artifact tab now empty for models, lifecycle decoupled from run, etc. Cites the [MLflow 3 migration page](https://mlflow.org/docs/latest/ml/mlflow-3/).
- **Outputs:** Two `LoggedModel`s (LR + LGBM); the LGBM `model_id` is saved to a workshop scratch table for Module 3 to pick up.

#### Module 3 — Hyperparameter Tuning + Evaluation
- **Folder:** `modules/03_tuning_and_eval/`
- **Runtime target:** ≤6 min
- **Concepts:** Optuna with MLflow, `mlflow.evaluate` for classification, custom business metric
- **APIs used:**
  - Optuna study with 15 trials, each trial logged as a nested `mlflow.start_run(nested=True)` — manual `log_params` + `log_metric` per trial
  - Final retrain with best params → new `LoggedModel` for `lgbm_tuned`
  - `mlflow.evaluate(model=lgbm_tuned_model, data=eval_df, model_type='classifier', targets='churned', extra_metrics=[expected_retention_value], model_id=tuned_model_id)` — Ref: [`mlflow.models.evaluate`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html). The `model_id=` param binds eval results to the LoggedModel — call out the lineage benefit
  - Custom metric via `mlflow.metrics.make_metric(eval_fn=expected_retention_value_fn, greater_is_better=True, name='expected_retention_value')` — Ref: [`mlflow.metrics`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html)
- **Business metric:** "Expected retention value" = (predicted churn probability × avg LTV × intervention success rate) summed over top-K targeted customers
- **Outputs:** A tuned LightGBM `LoggedModel` whose `model_id` is the input to Module 4

#### Module 4 — Registry & Model Serving (the runtime-critical module)
- **Folder:** `modules/04_registry_and_serving/`
- **Runtime target:** ≤8 min — endpoint provisioning is the dominant cost
- **Strategy:** Cell 1 kicks off endpoint creation in background; cells 2..N do registration + batch scoring while it provisions; final cells wait for endpoint readiness and query it
- **Concepts:** UC Model Registry, aliases (`@champion`/`@challenger`), Model Serving via SDK, batch scoring via `models:/`
- **APIs used:**
  - `mlflow.set_registry_uri("databricks-uc")` — Ref: [manage model lifecycle](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/)
  - `mlflow.register_model(model_uri=f"models:/{tuned_model_id}", name=f"{cat}.{sch}.bolttech_churn_model")`
  - `MlflowClient().set_registered_model_alias(name, "champion", version)` and `"challenger"` for the LR baseline
  - Batch scoring via `mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{cat}.{sch}.bolttech_churn_model@champion")`
  - Endpoint:
    ```python
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    w.serving_endpoints.create_and_wait(name=endpoint_name, config=EndpointCoreConfigInput(
        served_entities=[ServedEntityInput(
            name="churn-champion",
            entity_name=f"{cat}.{sch}.bolttech_churn_model",
            entity_version="<champion_version>",       # alias resolved here
            workload_size="Small",
            scale_to_zero_enabled=True)]))
    ```
    Ref: [Create custom serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints). **FLAG: re-verify whether `entity_version` accepts an alias literal directly; if not, resolve alias → version with `client.get_model_version_by_alias` first.**
  - Query:
    ```python
    import mlflow.deployments
    client = mlflow.deployments.get_deploy_client("databricks")
    client.predict(endpoint=endpoint_name, inputs={"dataframe_split": {...}})
    ```
- **Inference-table enablement:** see §6 fork on AI Gateway vs simulate

#### Module 5 — Production Monitoring
- **Folder:** `modules/05_monitoring/`
- **Runtime target:** ≤4 min
- **Concepts:** Inference-log monitoring, drift detection
- **APIs used (chosen path — see §8 fork):** New `WorkspaceClient.data_quality.create_monitor(...)` with `DataProfilingConfig(inference_log=InferenceLogConfig(problem_type=INFERENCE_PROBLEM_TYPE_CLASSIFICATION, ...))` — Ref: [Create a monitor via API](https://docs.databricks.com/aws/en/lakehouse-monitoring/create-monitor-api). Fallback path documented in code comment: `databricks.lakehouse_monitoring.create_monitor(...)`.
- **Approach:** Generate a small synthetic "inference table" with two time windows — baseline window (matches training distribution) and drifted window (shift `payment_failures_60d` distribution). Write to Delta. Create monitor. Query the auto-generated drift metric table to show drift on the shifted feature.
- **Output discussion:** A quick markdown block on real-deployment monitoring patterns (alert thresholds, retraining triggers, slicing by country).

### Part B — GenAI with MLflow 3

#### Module 6 — Tracing & Prompt Registry Fundamentals
- **Folder:** `modules/06_tracing_and_prompts/`
- **Runtime target:** ≤3 min
- **Concepts:** MLflow Tracing, autolog for LLM clients, Prompt Registry (register, version, alias, load)
- **APIs used:**
  - `mlflow.openai.autolog()` — Ref: [OpenAI integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai)
  - `from databricks_openai import get_openai_client; client = get_openai_client()` (or plain `openai.OpenAI(base_url=..., api_key=...)` using `dbutils.secrets`/token) — Ref: [Score foundation models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models)
  - `client.chat.completions.create(model="databricks-claude-haiku-4-5", messages=[...])` — trace appears in the experiment's Traces tab
  - `@mlflow.trace(name="...", span_type=SpanType.LLM)` decorator on a wrapper function — show manual tracing alongside autolog
  - `mlflow.genai.register_prompt(name="churn-summary-v1", template="Summarize churn drivers for {{customer_id}} given the following tickets:\n{{tickets}}\n\nAnswer in <= 100 words.")` — Ref: [Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/). Note: **double curly braces**.
  - `mlflow.genai.set_prompt_alias("churn-summary-v1", alias="production", version=1)`
  - `mlflow.genai.load_prompt("prompts:/churn-summary-v1@production")` → format + call LLM
- **Outputs:** A registered prompt with `@production` alias used by Modules 7–8.

#### Module 7 — RAG: "Why are customers churning?"
- **Folder:** `modules/07_rag_churn_insights/`
- **Runtime target:** ≤8 min — VS index sync is the dominant cost (~1 min on 500 rows but provisioning the endpoint, if cold, takes longer — kick off in Module 6 cell 1 or pre-provision)
- **Concepts:** Vector Search (Delta Sync, managed embeddings), retrieval, RAG chain composition, retrieval tracing
- **APIs used:**
  - `from databricks.vector_search.client import VectorSearchClient`
  - `client.create_endpoint(name=vs_endpoint, endpoint_type='STANDARD')` (idempotent — `try/except` on already-exists)
  - Source table: `support_tickets` augmented with `change_data_feed=true` (set in Module 0)
  - `client.create_delta_sync_index(endpoint_name=..., source_table_name=..., index_name=..., pipeline_type='TRIGGERED', primary_key='ticket_id', embedding_source_column='description', embedding_model_endpoint_name='databricks-gte-large-en')` — Ref: [Create & query Vector Search](https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search)
  - `client.get_index(...).similarity_search(query_text=..., columns=[...], num_results=5)`
  - RAG chain assembled as a traced Python function: `@mlflow.trace` on retrieve, augment, LLM-call steps
- **Outputs:** A VS index and a RAG callable used in Module 8.

#### Module 8 — Retention Outreach Agent (deploys for real — runtime-critical)
- **Folder:** `modules/08_retention_agent/`
- **Runtime target:** ≤9 min (raised from 5 → 9 per locked Q3; ~8–12 min agent endpoint cold-start absorbed in-cell)
- **Concepts:** `ResponsesAgent` authoring, tool definition, agent logging with `resources=` (auto-auth passthrough to Module 4 endpoint + Module 7 VS index), UC registration, `agents.deploy()` end-to-end
- **Files:**
  - `agent.py` — the agent module (logged via `python_model=agent.py` Models-from-Code pattern)
  - `08_retention_agent.py` — the driver notebook that imports, tests, logs, registers, deploys
- **Notebook structure** (key — absorbs cold-start productively):
  - **Cell 1:** Author + log agent + UC-register + kick off `agents.deploy()` *non-blocking* (so it provisions in background)
  - **Cells 2–N:** Walk through the agent architecture (markdown + code), load the logged agent locally via `models:/<model_id>`, exercise it on 2–3 sample customers, show traces in the UI
  - **Final cells:** Poll deploy status (a small `while` with a max-wait); once ready, hit the deployed endpoint with the same input and compare the response. If timeout hits, surface the endpoint URL and let participants pick up in Module 10.
- **APIs used:**
  - `from mlflow.pyfunc import ResponsesAgent` + `ResponsesAgentRequest`, `ResponsesAgentResponse` — Ref: [Author agent](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent), [ResponsesAgent](https://mlflow.org/docs/latest/genai/serving/responses-agent)
  - Inner-agent framework: **OpenAI Agents SDK** (`openai-agents` package; `from agents import Agent, function_tool`) — locked per Q1
  - Tools:
    - `get_customer_churn_score(customer_id: str)` — hits Module 4 endpoint via `mlflow.deployments.get_deploy_client("databricks").predict(...)`
    - `retrieve_customer_tickets(customer_id: str, query: str)` — filters VS index by `customer_id` then `similarity_search`
    - `draft_retention_email(customer_id, churn_score, ticket_themes) -> str` — pure prompt → LLM
  - Logging:
    ```python
    from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
    logged = mlflow.pyfunc.log_model(
        python_model="agent.py",
        name="retention_agent",
        input_example={"input": [{"role":"user","content":"Draft a retention email for customer CUST_000001"}]},
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-claude-haiku-4-5"),
            DatabricksServingEndpoint(endpoint_name=churn_endpoint),
            DatabricksVectorSearchIndex(index_name=vs_index),
        ],
        pip_requirements=["mlflow[databricks]>=3.12","databricks-openai","openai-agents"],
    )
    ```
  - UC registration: `mlflow.set_registry_uri("databricks-uc"); uc_info = mlflow.register_model(model_uri=logged.model_uri, name=AGENT_MODEL_NAME)`
  - Deploy (non-blocking-ish; the SDK call itself returns quickly but the endpoint takes time to become READY):
    ```python
    from databricks import agents
    deployment = agents.deploy(AGENT_MODEL_NAME, uc_info.version, scale_to_zero_enabled=True)
    # deployment.endpoint_url, .review_app_url, .query_endpoint
    ```
    Ref: [Deploy agent](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent)
  - Local test (during cold-start): load via `mlflow.pyfunc.load_model(f"models:/{logged.model_id}")` and call `.predict({"input":[{"role":"user","content":"..."}]})` — fast
- **Module 9 dependency note:** M9 evaluates the **locally-loaded** agent (faster, doesn't depend on deploy timing). M10 hits the **deployed** endpoint to demonstrate the full production path.

#### Module 9 — GenAI Evaluation
- **Folder:** `modules/09_genai_evaluation/`
- **Runtime target:** ≤6 min
- **Concepts:** `mlflow.genai.evaluate`, built-in scorers, custom `Guidelines`, prompt iteration loop
- **APIs used:**
  - Eval dataset: ~25 hand-curated examples, `inputs={"customer_id": "...", "question": "..."}`, `expectations={"expected_facts": [...], "guidelines": "..."}`
  - `predict_fn` wraps the local-loaded agent's `.predict` so the eval can unpack `inputs` as kwargs
  - `mlflow.genai.evaluate(data=eval_df, predict_fn=predict_fn, scorers=[Correctness(), Safety(), RetrievalGroundedness(), Guidelines(name="bolttech_voice", guidelines="The email must mention the customer's plan tier and country; the tone must be warm but professional; never promise a specific discount.")])` — Ref: [Predefined judge scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/), [Agent eval migration](https://docs.databricks.com/aws/en/mlflow3/genai/agent-eval-migration)
  - Iteration: tweak the registered prompt template → bump version (auto on `register_prompt`) → re-run eval → compare runs in UI
- **Outputs:** Two evaluation runs (v1 vs v2 prompt) so participants see the loop work end-to-end.

#### Module 10 — Capstone: Putting It Together
- **Folder:** `modules/10_capstone/`
- **Runtime target:** ≤5 min
- **Concepts:** End-to-end orchestration, productionization discussion
- **APIs used:** All of the above stitched together — no new APIs
- **Flow:**
  1. Spark-UDF batch-score the current customer snapshot via `models:/...@champion`
  2. Rank top-10 highest churn risk
  3. For each, invoke the locally-loaded agent → drafted email
  4. Display results in a styled table
  5. Markdown block on productionization: Jobs, scheduled workflow, human-in-the-loop review (Databricks Review App), Slack alerting

---

## 4. Runtime Budget Allocation

| Module | Target (prompt) | My estimate | Notes / risk |
|---|---|---|---|
| 00 Setup | ≤3 min | ~2 min | Synthetic gen is in-memory then Delta write |
| 01 Feature engineering | ≤4 min | ~3 min | Small data; UC writes fast |
| 02 Experiment tracking | ≤6 min | ~5 min | LR + LGBM both fast on 20k rows |
| 03 Tuning & eval | ≤6 min | ~5 min | 15 Optuna trials × ~10s each |
| 04 Registry & serving | ≤8 min | ~7 min | Endpoint cold-start is the unknown; mitigated by background-provisioning pattern |
| 05 Monitoring | ≤4 min | ~3 min | Monitor creation is async; show drift metrics from a small-but-instant simulated run |
| 06 Tracing & prompts | ≤3 min | ~2 min | Single chat call + prompt register |
| 07 RAG | ≤8 min | ~6 min | VS endpoint pre-provisioned (kicked off in M6); Delta Sync TRIGGERED ~1 min for 500 rows |
| 08 Agent | ≤9 min (revised) | ~8 min | Background `agents.deploy()` + local-test absorbs the cold-start; final cells query the live endpoint |
| 09 GenAI eval | ≤6 min | ~5 min | 25 examples × ~4 scorers = ~100 judge calls; warn about OTPM if shared workspace; uses local agent (no dep on M8 deploy timing) |
| 10 Capstone | ≤5 min | ~4 min | Pure orchestration; uses the deployed M8 agent endpoint |
| **Total** | **≤62 min** | **~50 min** | ~10 min headroom for narration & Q&A; ~25 min slack vs 60 min hard cap |

**Risk hotspots (will be explicit in `VERIFICATION.md`):**
- Module 4 endpoint cold-start (no SLA in docs; my estimate is 5–10 min on a small workload)
- Module 7 VS endpoint provisioning if cold (kicked off in Module 6 to absorb the wait)
- Module 9 if eval-judge throughput hits FMAPI OTPM (200K input, 20K output per workspace)

---

## 5. Cross-module Coherence (threading)

These continuity invariants will be enforced and re-checked in the cross-cutting pass (task #14):

- **One config file:** `config/workshop_config.py` is the single source of truth for `CATALOG`, `SCHEMA`, `MODEL_NAME`, `SERVING_ENDPOINT_NAME`, `VS_ENDPOINT_NAME`, `VS_INDEX_NAME`, `AGENT_MODEL_NAME`, `CHAT_MODEL`, `EMBEDDING_MODEL`. Every notebook imports from this.
- **`customer_id` is a `STRING` everywhere.** Decided here once to avoid joins breaking on type coercion.
- **The LGBM `LoggedModel` from Module 2** → Module 3 retrains a tuned variant → Module 4 registers the tuned variant → Module 5 monitors it → Module 8 agent calls Module 4's endpoint.
- **Module 6's `@production` prompt alias** is loaded by Module 7's RAG chain and Module 8's agent.
- **Module 7's VS index** is the retrieval tool inside Module 8's agent.
- **Module 9's `predict_fn`** wraps the same locally-loaded agent that Module 8 logged — not a re-implementation.
- **Module 10** uses the same endpoint name, agent, and VS index identifiers — no copy-paste drift.

---

## 6. Repo Layout

```
mlflow3-databricks-churn-workshop/
├── README.md                          # Top-level (per §7 of the prompt)
├── PLAN.md                            # This file
├── LICENSE                            # Apache 2.0
├── .gitignore                         # Python + Databricks artifacts
├── CONTRIBUTING.md                    # Brief
├── requirements.txt                   # Pinned MLflow 3.12, databricks-* libs, optuna, lightgbm
├── config/
│   └── workshop_config.py             # Single source of truth (see §5)
├── setup/
│   └── 00_setup_and_synthetic_data.py # Databricks notebook source
├── modules/
│   ├── 01_feature_engineering/
│   │   ├── 01_feature_engineering.py
│   │   └── README.md
│   ├── 02_experiment_tracking/
│   │   ├── 02_experiment_tracking.py
│   │   └── README.md
│   ├── 03_tuning_and_eval/
│   │   ├── 03_tuning_and_eval.py
│   │   └── README.md
│   ├── 04_registry_and_serving/
│   │   ├── 04_registry_and_serving.py
│   │   └── README.md
│   ├── 05_monitoring/
│   │   ├── 05_monitoring.py
│   │   └── README.md
│   ├── 06_tracing_and_prompts/
│   │   ├── 06_tracing_and_prompts.py
│   │   └── README.md
│   ├── 07_rag_churn_insights/
│   │   ├── 07_rag_churn_insights.py
│   │   └── README.md
│   ├── 08_retention_agent/
│   │   ├── 08_retention_agent.py      # Driver notebook
│   │   ├── agent.py                   # ResponsesAgent subclass, logged via Models-from-Code
│   │   └── README.md
│   ├── 09_genai_evaluation/
│   │   ├── 09_genai_evaluation.py
│   │   ├── eval_dataset.py            # 25-example eval set as a Python module (importable)
│   │   └── README.md
│   └── 10_capstone/
│       ├── 10_capstone.py
│       └── README.md
├── docs/
│   ├── architecture.md                # Mermaid diagrams
│   ├── instructor_guide.md            # Pacing, common pitfalls, talking points
│   ├── research_log.md                # Dated URL log per prompt §7.2
│   └── images/
├── scripts/
│   └── reset_workshop.py              # Tear down catalog/schema/endpoints/indexes/registered model
└── VERIFICATION.md                    # Self-review report per prompt §7.6
```

Two structural additions beyond the prompt's proposed layout, callouts so you can push back:
- `modules/08_retention_agent/agent.py` — a separate file rather than the agent class inline in the notebook. This is required by `mlflow.pyfunc.log_model(python_model="agent.py", ...)` (Models-from-Code). Cleaner serialization and matches the current Databricks template.
- `modules/09_genai_evaluation/eval_dataset.py` — eval examples as a Python module (a `list[dict]`) rather than a separate Delta/JSON file. Lets the notebook import + display in one cell and keeps the dataset under version control.

---

## 7. Process & Verification Plan

Per the prompt's §7:

1. **Plan** ← this document
2. **Scaffold** — directory tree, stub READMEs, top-level files (only after you approve §1–§8)
3. **Build modules 0 → 10 in dependency order**, with a self-review pass after each
4. **Cross-cutting coherence pass** to enforce §5 invariants
5. **Verification report (`VERIFICATION.md`)** — per-notebook checklist:
   - ✓ Every MLflow / Databricks API verified against a live docs URL (cited inline as `# Ref: <url>`)
   - ✓ Idempotent on re-run (no "already exists" crashes)
   - ✓ Reads from `workshop_config.py`, no hardcoded names
   - ✓ Header cell with title / objectives / prerequisites / runtime / compute
   - ✓ Footer with recap + go-deeper links + handoff
   - ✓ Best-estimate runtime vs target; flag any over-budget
   - ✗ Anything I couldn't verify without a live workspace will be flagged explicitly (e.g., exact `entity_version`-accepts-alias behavior, AI Gateway inference table field names)

6. **Research log (`docs/research_log.md`)** — every URL consulted, dated, with one-line note. Pre-seeded with the 30+ URLs from this research pass.

---

## 8. Open Questions — RESOLVED 2026-05-25

Locked answers preserved here for traceability:
- **Q1** → OpenAI Agents SDK
- **Q2** → Per-user schemas (`bolttech_workshop.churn_<sanitized_user>`)
- **Q3** → **Actually deploy** via `agents.deploy()` (Module 8 runtime bumped to ≤9 min — see §3 and §4)
- **Q4** → `bolttech_workshop` (default; no override)
- **Q5** → Zero-padded strings `CUST_000001` (default; no override)
- **Q6** → Simulated inference table with synthetic drift

(Original wording preserved below for record.)


**Q1 — Inner-agent framework.** Inside the `ResponsesAgent.predict(...)` body, which framework should drive the tool-loop?

| Option | Pros | Cons |
|---|---|---|
| **OpenAI Agents SDK** (`openai-agents` package) (default) | Less boilerplate; Databricks' own template uses it; OpenAI-Responses-API-shaped already | One more pip dep; less popular than LangGraph in some Databricks customer codebases |
| LangGraph | Explicit graph, customers may already use it | More code; you need to manually shape outputs to `ResponsesAgent` schema |

**Q2 — Catalog/schema strategy.**

| Option | Pros | Cons |
|---|---|---|
| **Per-user schemas** `bolttech_workshop.churn_<sanitized_user>` (default) | Safe for shared workspaces; no collisions on shared serving endpoint names either (also per-user) | Slightly more setup boilerplate |
| Single shared `bolttech_workshop.churn` | Simpler config; cleaner names | Race conditions if multiple participants run at once on shared workspace |

**Q3 — Agent endpoint deployment in workshop.**

| Option | Pros | Cons |
|---|---|---|
| **Log + test locally; show deploy as text only** (default) | Module 8 stays under 5 min budget | Participants don't see the agent as a deployed endpoint live |
| Actually deploy via `agents.deploy()` | Full production demo | Adds ~10 min cold-start; would need to move it to Module 8 cell 1 and use background-provisioning pattern, blows module budget |

**Q4 — Workshop catalog name.** I'll default to `bolttech_workshop`. Override if you want something different (`mlflow3_workshop`, `fe_demos_bolttech`, etc.).

**Q5 — `customer_id` format.** I'll default to zero-padded strings like `"CUST_000001"` for readability in agent outputs. Push back if you'd prefer integers or UUIDs.

**Q6 — Anything in the curriculum you'd cut or reframe?** My main reservation: Module 5 (monitoring) is lightweight in 4 minutes. I'll demo on a simulated inference table — call out if you'd rather demo on real endpoint traffic (would require participants to send several hundred predictions to the endpoint to populate the inference table, adding ~2 min).

---

## 9. Citation Index — Key Docs URLs

(Will be expanded in `docs/research_log.md`. These are the ~30 sources my research consulted.)

### MLflow 3
- https://mlflow.org/docs/latest/ — landing
- https://mlflow.org/docs/latest/ml/mlflow-3/ — migration / breaking changes
- https://mlflow.org/docs/latest/ml/tracking/autolog/ — autolog supported flavors
- https://mlflow.org/docs/latest/ml/model-registry/workflow/ — alias workflow
- https://mlflow.org/docs/latest/ml/evaluation/ — `mlflow.evaluate`
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html
- https://mlflow.org/docs/latest/genai/prompt-registry/
- https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/
- https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/
- https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/
- https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/supported-models/
- https://mlflow.org/docs/latest/genai/tracing/
- https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/
- https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai
- https://mlflow.org/docs/latest/genai/flavors/chat-model-intro/
- https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro/
- https://mlflow.org/docs/latest/genai/serving/responses-agent

### Databricks platform
- https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml — DBR ML LTS
- https://docs.databricks.com/aws/en/compute/serverless/dependencies — Serverless ML base env
- https://docs.databricks.com/aws/en/compute/serverless/limitations
- https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc
- https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series — point-in-time
- https://docs.databricks.com/aws/en/machine-learning/feature-store/online-tables
- https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints
- https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints
- https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models
- https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/
- https://docs.databricks.com/aws/en/lakehouse-monitoring/create-monitor-api
- https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models
- https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits
- https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search
- https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent
- https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent
- https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent
- https://docs.databricks.com/aws/en/mlflow3/genai/agent-eval-migration
- https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers
- https://docs.databricks.com/aws/en/ai-gateway/

### Flagged for re-verify before writing code (also in `VERIFICATION.md`)
- `mlflow.get_logged_model` exact signature (search-snippet evidence; live API page truncated)
- `mlflow.create_logged_model` / `mlflow.set_active_model` existence + signatures
- Built-in scorer constructor signatures — whether they accept per-instance `model=` kwarg
- `ServedEntityInput.entity_version` — does it accept an alias literal or require numeric version?
- AI Gateway `AiGatewayInferenceTableConfig` field names
- Lakehouse Monitoring new `WorkspaceClient.data_quality` vs legacy `databricks.lakehouse_monitoring` — pick exactly one and stick to it
- `agents.deploy()` full signature
- OpenAI Agents SDK package name (`openai-agents` vs `agents`) and import path
- Whether DBR Serverless ML (Beta) has the MLflow 3.12 features or only DBR 17.3 ML LTS does

---

## 10. What I'll do next (waiting on your sign-off)

When you approve / push back on §8:
1. Scaffold the tree per §6 with all stubs in place (~5 min of work).
2. Build Module 0 (synthetic data) — verify it runs in a Databricks notebook source format check.
3. Continue 1 → 10, with a self-review pass after each.
4. Coherence pass (§5 invariants).
5. `VERIFICATION.md` + top-level README + architecture diagram + instructor guide.

Estimated total build time (my side): a few hours of focused work. I'll summarize after each meaningful chunk (plan, scaffold, each module, final verification) per your "what I want back" section.
