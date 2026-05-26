# Research Log

Every doc URL consulted during the build of this workshop, dated, with a one-line note. This is the citation appendix referenced from `PLAN.md` §9 and from inline `# Ref: <url>` comments in the notebooks.

> Source-of-truth date for all entries below: docs were live as of **2026-05-24** unless otherwise noted. Re-verify before any cited API call if the docs may have moved.

## MLflow 3 — core / classic ML

| Date | URL | Note |
|---|---|---|
| 2026-05-24 | https://mlflow.org/docs/latest/ | Latest docs landing TOC |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/ | Classic-ML root |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/mlflow-3/ | MLflow 3 migration / breaking changes — `artifact_path` → `name`, `baseline_model` removed, etc. |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/tracking/ | Tracking root; `models:/<model_id>` URI scheme |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/tracking/autolog/ | `mlflow.autolog` 3.x signature + supported flavors (sklearn, LightGBM, XGBoost, etc.) — `log_traces=True` default |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/evaluation/ | `mlflow.evaluate` for classification + custom metrics |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/model-registry/ | UC registry pointer (`mlflow.set_registry_uri("databricks-uc")`) |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/model-registry/workflow/ | Alias API — `set_registered_model_alias` |
| 2026-05-24 | https://mlflow.org/docs/latest/ml/model/ | LoggedModel entity / Models page |
| 2026-05-24 | https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html | Top-level API (partial render; some methods like `get_logged_model` flagged for re-verify) |
| 2026-05-24 | https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html | `mlflow.sklearn.log_model` 3.x signature |
| 2026-05-24 | https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html | `mlflow.models.evaluate` signature |
| 2026-05-24 | https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html | `mlflow.metrics.make_metric` for custom business metrics |
| 2026-05-24 | https://github.com/mlflow/mlflow/releases | Release notes — current 3.12.0 (verify exact date when citing) |

## MLflow 3 — GenAI

| Date | URL | Note |
|---|---|---|
| 2026-05-24 | https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html | Full `mlflow.genai` module map |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/ | GenAI section root |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/prompt-registry/ | `register_prompt`, `{{var}}` template syntax (double curly), versioning |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/ | `set_prompt_alias`, `load_prompt("prompts:/name@alias")` |
| 2026-05-24 | https://mlflow.org/docs/latest/api_reference/_modules/mlflow/genai/evaluation/base.html | Source for `mlflow.genai.evaluate(data, scorers, predict_fn, model_id)` — 4 params only |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/ | Built-in scorers: `Correctness`, `Safety`, `Guidelines`, `RetrievalGroundedness`, etc. |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/ | Custom `@scorer` decorator + `Feedback` return type |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/supported-models/ | Judge model URI formats — `"databricks:/databricks-claude-sonnet-4-5"`, `"openai:/gpt-4o-mini"`, etc. |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/concepts/scorers/ | Scorer concepts |
| 2026-05-24 | https://mlflow.org/docs/latest/api_reference/_modules/mlflow/genai/judges/make_judge.html | `make_judge(name, instructions, model, feedback_value_type, ...)` |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/tracing/ | Tracing root (GenAI-oriented) |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/ | `@mlflow.trace(name=..., span_type=SpanType.LLM)` decorator |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/tracing/integrations/ | Auto-tracing integrations index — OpenAI, Anthropic, LangChain, DSPy, LangGraph, etc. |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai | `mlflow.openai.autolog()` |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/flavors/ | Flavors index for GenAI logging |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/flavors/chat-model-intro/ | `ChatModel` (OpenAI-compatible chat) |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro/ | `ResponsesAgent` introduction |
| 2026-05-24 | https://mlflow.org/docs/latest/genai/serving/responses-agent | `ResponsesAgent` interface + `predict`/`predict_stream` |

## Databricks platform — classic ML

| Date | URL | Note |
|---|---|---|
| 2026-05-24 | https://docs.databricks.com/aws/en/release-notes/runtime/ | DBR release-notes index |
| 2026-05-24 | https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml | DBR 17.3 LTS ML — Spark 4, MLflow 3.0.1, sklearn 1.6.1, LightGBM 4.6.0, XGBoost 3.0.0, Optuna 3.6.1, `databricks-feature-engineering` 0.12.1 |
| 2026-05-24 | https://docs.databricks.com/aws/en/release-notes/runtime/16.4lts-ml | DBR 16.4 LTS ML — fallback if 17.3 unavailable |
| 2026-05-24 | https://docs.databricks.com/aws/en/compute/serverless/dependencies | Serverless ML base env — pip / `%uv pip` / Environment panel |
| 2026-05-24 | https://docs.databricks.com/aws/en/compute/serverless/limitations | Serverless limitations — no init scripts / cluster libs / JARs; 1 GB UDF memory |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc | `FeatureEngineeringClient.create_table`/`write_table` — UC feature tables |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series | `FeatureLookup(..., timestamp_lookup_key=...)` for point-in-time joins |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/feature-store/online-tables | New `create_online_store` + `publish_table` flow (replaces legacy Online Tables) |
| 2026-05-24 | https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html | `FeatureEngineeringClient` API reference |
| 2026-05-24 | https://api-docs.databricks.com/python/feature-engineering/latest/ml_features.feature_lookup.html | `FeatureLookup` API reference |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints | `WorkspaceClient.serving_endpoints.create_and_wait(...)` |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints | Invocation — `mlflow.deployments.get_deploy_client("databricks").predict(...)` |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables | Legacy inference tables — deprecating (forces AI Gateway path going forward) |
| 2026-05-24 | https://docs.databricks.com/aws/en/ai-gateway/inference-tables | AI Gateway inference tables (UI only on this page; SDK fields flagged for re-verify) |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/ | UC Model Registry workflow — three-part names, alias API, permissions |
| 2026-05-24 | https://docs.databricks.com/aws/en/mlflow/models | UC models page (Databricks-side) |
| 2026-05-24 | https://docs.databricks.com/aws/en/lakehouse-monitoring/ | Lakehouse Monitoring overview |
| 2026-05-24 | https://docs.databricks.com/aws/en/lakehouse-monitoring/create-monitor-api | `WorkspaceClient.data_quality.create_monitor` with `InferenceLogConfig` |
| 2026-05-24 | https://docs.databricks.com/aws/en/lakehouse-monitoring/monitor-output | Output metric tables (profile + drift) |
| 2026-05-24 | https://databricks-sdk-py.readthedocs.io/en/latest/workspace/serving/serving_endpoints.html | Python SDK reference for `serving_endpoints` |

## Databricks platform — GenAI

| Date | URL | Note |
|---|---|---|
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/ | FMAPI overview |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models | Pay-per-token chat + embedding models — `databricks-claude-haiku-4-5`, `databricks-gte-large-en`, etc. |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits | Rate limits — 200K ITPM / 20K OTPM / 360K QPH per workspace |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models | `databricks-openai` client; `openai.OpenAI(base_url=...)` pattern |
| 2026-05-24 | https://docs.databricks.com/aws/en/machine-learning/model-serving/query-openai-responses | OpenAI Responses API on Databricks |
| 2026-05-24 | https://docs.databricks.com/aws/en/generative-ai/vector-search | Vector Search overview |
| 2026-05-24 | https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search | `VectorSearchClient` — `create_endpoint`, `create_delta_sync_index`, `similarity_search` |
| 2026-05-24 | https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent | **`ResponsesAgent` is the canonical 2026 pattern** — ChatAgent retired |
| 2026-05-24 | https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent | `mlflow.pyfunc.log_model(python_model="agent.py", resources=[...])` — Models-from-Code |
| 2026-05-24 | https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent | `databricks.agents.deploy(uc_name, version)` — provisions endpoint + Review App + inference tables |
| 2026-05-24 | https://docs.databricks.com/aws/en/generative-ai/tutorials/agent-quickstart | Agent quickstart tutorial |
| 2026-05-24 | https://docs.databricks.com/aws/en/mlflow3/genai/agent-eval-migration | Agent Eval migrated into `mlflow.genai.evaluate` |
| 2026-05-24 | https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers | Scorers concepts (Databricks-side) |
| 2026-05-24 | https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/eval-examples | `mlflow.genai.evaluate` examples |
| 2026-05-24 | https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/predefined-judge-scorers | Predefined judge scorers (Databricks docs view) |
| 2026-05-24 | https://docs.databricks.com/aws/en/ai-gateway/ | Unity AI Gateway (Beta) — closing-slide mention only |

## Flagged for re-verify before writing the cited code

Tracked in `VERIFICATION.md`:

- `mlflow.get_logged_model` exact signature (live API page truncated)
- `mlflow.create_logged_model` / `mlflow.set_active_model` existence + signatures
- Whether built-in scorer constructors accept per-instance `model=` kwarg
- Whether `ServedEntityInput.entity_version` accepts an alias literal or requires resolution
- AI Gateway `AiGatewayInferenceTableConfig` exact field names
- Pick **one** of `WorkspaceClient.data_quality` vs `databricks.lakehouse_monitoring` and stick to it
- `agents.deploy()` full signature (tags, env vars, workload size)
- OpenAI Agents SDK pip name + import path (`openai-agents` vs `agents`)
- MLflow 3.12 features on Serverless base env vs DBR 17.3 LTS ML only

This log is appended to throughout the build — new URLs land here with a date as they're consulted.
