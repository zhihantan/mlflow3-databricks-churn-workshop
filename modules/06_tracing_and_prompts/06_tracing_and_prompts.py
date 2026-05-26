# Databricks notebook source
# MAGIC %md
# MAGIC # Module 06 — Tracing & Prompt Registry Fundamentals
# MAGIC ### Observability + version control for LLM calls — the GenAI foundation
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > **TL;DR** — Enable `mlflow.openai.autolog()`, make a chat call against Databricks Foundation Model APIs, and every input/output/latency/token count is auto-captured as an MLflow Trace. Then register a `{{var}}`-templated prompt with an `@production` alias — every downstream module loads it by alias.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Welcome to the **GenAI half** of the workshop. We start with two foundational MLflow 3 primitives that everything else builds on:
# MAGIC
# MAGIC 1. **MLflow Tracing** — every LLM call gets captured as a structured trace (inputs, outputs, latency, tokens, errors) viewable in the experiment's *Traces* tab.
# MAGIC 2. **MLflow Prompt Registry** — versioned, named prompt templates with aliases (`@production`, etc.) and a `{{variable}}` template syntax. Loadable from any notebook by name + alias.
# MAGIC
# MAGIC We also kick off the **Vector Search endpoint** in cell 1 so Module 7 finds it ready (same background-provisioning pattern as Module 4's serving endpoint).
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Make a chat call against the Databricks Foundation Model APIs via the OpenAI-compatible client.
# MAGIC - See the call automatically traced into the MLflow Traces tab via `mlflow.openai.autolog()`.
# MAGIC - Decorate a wrapper function with `@mlflow.trace` for custom span naming.
# MAGIC - Register a prompt template, set a `@production` alias, and load + format it by alias.
# MAGIC
# MAGIC **Databricks features showcased**
# MAGIC
# MAGIC - **Databricks Foundation Model APIs (FMAPI)** — pay-per-token access to `databricks-claude-haiku-4-5`, `databricks-claude-sonnet-4-6`, `databricks-meta-llama-3-3-70b-instruct`, `databricks-gpt-5-mini`, and embedding models like `databricks-gte-large-en`. Same workspace, same auth, same governance as your data — no separate vendor account or BYOK plumbing.
# MAGIC - **OpenAI-compatible client against FMAPI** — point a plain `openai.OpenAI` client at `{workspace}/serving-endpoints` with your workspace token; every OpenAI SDK pattern you already know works unchanged. No vendor lock-in either direction.
# MAGIC - **MLflow Tracing** (`mlflow.openai.autolog()`) — one line patches the OpenAI SDK to emit a full trace per call: inputs, outputs, token counts, latency, errors. Visible in the MLflow Traces tab in the experiment UI; queryable via `mlflow.search_traces(...)`.
# MAGIC - **`@mlflow.trace` decorator with `SpanType.CHAIN` / `SpanType.LLM`** — explicit named spans wrap auto-traced calls so the trace tree shows your pipeline structure (retrieve → augment → generate), not just one flat LLM call.
# MAGIC - **MLflow Prompt Registry** — versioned prompt templates with `{{variable}}` Jinja-style syntax, aliases (`@production`), full lifecycle on the platform. `mlflow.genai.register_prompt(...)` creates an immutable version; `set_prompt_alias` swaps which version `@production` points at — every consumer that loads `prompts:/<name>@production` picks up the new version automatically.
# MAGIC - **Vector Search endpoint provisioning** (kicked off here for Module 7) — `VectorSearchClient.create_endpoint(...)` returns immediately; provisioning happens async in the background.
# MAGIC
# MAGIC **Why this matters for insurtech**
# MAGIC
# MAGIC Regulated insurers cannot ship customer-facing LLM features without (1) a defensible audit trail of every model call, (2) version-controlled prompts that survive personnel turnover, and (3) compliance teams able to inspect what the model actually said in any given interaction. MLflow Tracing + Prompt Registry give bolttech all three out of the box, in the same governance plane as their structured data. The alternative — duct-taping Langfuse + Helicone + GitHub-stored prompts + bespoke audit pipelines — is fragile and audit-hostile.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Module 0 has been run (we need the per-user catalog/schema to exist).
# MAGIC
# MAGIC **Expected runtime**: ~3 minutes.
# MAGIC
# MAGIC **Compute**: Serverless or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "openai>=1.50" \
# MAGIC   "databricks-vectorsearch>=0.50" \
# MAGIC   "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Imports & config

# COMMAND ----------

import os
import sys

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

from config.workshop_config import (  # noqa: E402
    CHAT_MODEL,
    EXPERIMENT_PATH,
    SUMMARY_PROMPT_NAME,
    VS_ENDPOINT,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Kick off Vector Search endpoint creation (background)
# MAGIC
# MAGIC Provisioning a VS endpoint takes a few minutes; firing it here gives Module 7 a head start. Idempotent — skips if the endpoint already exists.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing_endpoints = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
if VS_ENDPOINT in existing_endpoints:
    print(f"VS endpoint {VS_ENDPOINT} already exists; skipping.")
else:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"Kicked off VS endpoint creation: {VS_ENDPOINT} — will be ready by Module 7")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Enable tracing for the OpenAI client
# MAGIC
# MAGIC `mlflow.openai.autolog()` patches the OpenAI SDK so every `client.chat.completions.create(...)` (and several other methods) automatically produces an MLflow trace.
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai

# COMMAND ----------

import mlflow

# UC-backed Prompt Registry — required for register_prompt() to accept the 3-part
# `<catalog>.<schema>.<prompt_name>` identifiers defined in workshop_config.py.
mlflow.set_registry_uri("databricks-uc")

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.openai.autolog()
print(f"Tracing enabled. Traces will land in: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Build the OpenAI client targeting Databricks FMAPI
# MAGIC
# MAGIC The Databricks Foundation Model APIs are OpenAI-compatible — point a plain `openai.OpenAI` client at the workspace's `/serving-endpoints` URL and use the workspace API token. No extra packages required.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models

# COMMAND ----------

from openai import OpenAI

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
workspace_host = ctx.apiUrl().get()
workspace_token = ctx.apiToken().get()

client = OpenAI(
    api_key=workspace_token,
    base_url=f"{workspace_host}/serving-endpoints",
)
print(f"OpenAI client pointed at: {workspace_host}/serving-endpoints")
print(f"Default model: {CHAT_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. First traced chat call
# MAGIC
# MAGIC A plain chat call. Open the experiment's **Traces** tab afterwards to see the captured span — inputs, output, model name, token counts, latency, all in one place.

# COMMAND ----------

resp = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[
        {"role": "system", "content": "You are a concise customer-success assistant for an insurtech company."},
        {"role": "user", "content": "In one sentence, why do insurance customers most commonly churn?"},
    ],
    max_tokens=120,
)
print("Model response:")
print(resp.choices[0].message.content)
print(f"\nTokens — prompt: {resp.usage.prompt_tokens}, completion: {resp.usage.completion_tokens}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Add manual tracing for wrapper functions
# MAGIC
# MAGIC `@mlflow.trace` adds a named span around any Python function — useful when an LLM call sits inside a larger pipeline (retrieval, post-processing, etc.) and you want the trace tree to show those layers explicitly.
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/

# COMMAND ----------

from mlflow.entities import SpanType


@mlflow.trace(name="summarize_for_customer", span_type=SpanType.CHAIN)
def summarize_churn_for_customer(customer_summary: str) -> str:
    """A trivially-thin wrapper around a chat call, decorated to show as its own span."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Summarize the churn signal for an insurtech CS team in 2-3 sentences."},
            {"role": "user", "content": customer_summary},
        ],
        max_tokens=180,
    )
    return resp.choices[0].message.content


summary = summarize_churn_for_customer(
    "Customer CUST_000123 (SG, basic plan, 6mo tenure): 3 payment failures in last 60d, 1 pending claim, 2 negative-sentiment support tickets in last 30d."
)
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7. Register a prompt template in the Prompt Registry
# MAGIC
# MAGIC The Prompt Registry stores versioned prompt templates by name. Two big gotchas vs. MLflow 2 / vs. f-strings:
# MAGIC
# MAGIC 1. **Double curly braces** `{{var}}` — not single. Single braces would conflict with Jinja2 control flow.
# MAGIC 2. **Versions are immutable**. Re-registering the same name creates a new version, not a mutation.
# MAGIC
# MAGIC Ref: https://mlflow.org/docs/latest/genai/prompt-registry/

# COMMAND ----------

PROMPT_TEMPLATE_V1 = """You are an insurtech retention analyst at bolttech.

Given the following snapshot of a single customer, identify the top 1-2 churn drivers and recommend an outreach approach. Be concrete and concise. Avoid generic platitudes.

Customer ID: {{customer_id}}
Country: {{country}}
Plan tier: {{plan_tier}}
Recent support tickets:
{{tickets}}

Respond in <= 120 words. Structure:
- Drivers: (1-2 bullets)
- Suggested outreach: (1-2 sentences)
"""

prompt_v1 = mlflow.genai.register_prompt(
    name=SUMMARY_PROMPT_NAME,
    template=PROMPT_TEMPLATE_V1,
    commit_message="Initial churn-driver summary prompt (workshop)",
    tags={"task": "churn_summary", "model_target": CHAT_MODEL},
)
print(f"Registered prompt: {SUMMARY_PROMPT_NAME} version {prompt_v1.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 8. Set the `@production` alias

# COMMAND ----------

# Ref: https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/
mlflow.genai.set_prompt_alias(
    name=SUMMARY_PROMPT_NAME,
    alias="production",
    version=prompt_v1.version,
)
print(f"Alias set: {SUMMARY_PROMPT_NAME}@production → version {prompt_v1.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 9. Load by alias and use the prompt
# MAGIC
# MAGIC `mlflow.genai.load_prompt("prompts:/<name>@<alias>")` is the version-agnostic loader. Modules 7 and 8 will load this same prompt by alias so any future prompt iteration in Module 9 propagates without code changes.

# COMMAND ----------

loaded_prompt = mlflow.genai.load_prompt(f"prompts:/{SUMMARY_PROMPT_NAME}@production")

filled = loaded_prompt.format(
    customer_id="CUST_000123",
    country="Singapore",
    plan_tier="basic",
    tickets=(
        "1. 'My monthly premium increased without notice...' (negative)\n"
        "2. 'Claim CLM_001234 has been pending for 22 days...' (negative)\n"
        "3. 'Can't log into the bolttech app on my Android...' (neutral)"
    ),
)

print("--- Formatted prompt ---")
print(filled)
print("\n--- LLM response ---")
resp2 = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": filled}],
    max_tokens=250,
)
print(resp2.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 10. Verify everything by listing recent traces
# MAGIC
# MAGIC The experiment Traces tab is the UI view, but you can also query traces programmatically — useful for evaluation pipelines (Module 9).

# COMMAND ----------

# Ref: https://mlflow.org/docs/latest/genai/tracing/
# `locations=` replaces the deprecated `experiment_ids=` in MLflow 3. It expects
# experiment *IDs* (or other resource location URIs) — not experiment paths. A
# raw path here gets interpreted as a telemetry-profile UUID and 404s, so we
# still resolve the experiment to its ID first.
_experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id
recent_traces = mlflow.search_traces(locations=[_experiment_id], max_results=10)
print(f"Recent traces in this experiment: {len(recent_traces)}")
# The Traces tab in the MLflow experiment UI is the right place to inspect
# request/response payloads — they contain nested dicts that don't round-trip
# through Arrow, so a Databricks `display()` here would error. The count above
# is enough to confirm tracing is wired up.
if len(recent_traces):
    print(f"Most recent trace_id: {recent_traces.iloc[0]['trace_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just learned**
# MAGIC
# MAGIC - `mlflow.openai.autolog()` auto-traces every OpenAI SDK call. Same one-liner exists for `mlflow.anthropic`, `mlflow.langchain`, `mlflow.dspy`, and many more (see [tracing integrations](https://mlflow.org/docs/latest/genai/tracing/integrations/)).
# MAGIC - `@mlflow.trace(name=..., span_type=...)` adds custom span layers around your wrapper functions.
# MAGIC - The Prompt Registry uses `{{variable}}` template syntax (double curly braces!) and versions every registration.
# MAGIC - Aliases (`@production`) decouple downstream code from prompt versions — flip the alias, every consumer follows.
# MAGIC - Calls through the Databricks Foundation Model APIs use a plain `openai.OpenAI` client against `{workspace}/serving-endpoints` — no extra SDK needed.
# MAGIC
# MAGIC **What you'd build without Databricks**
# MAGIC
# MAGIC | Concern | DIY stack | Databricks-native |
# MAGIC | --- | --- | --- |
# MAGIC | LLM provider | OpenAI/Anthropic vendor account + key vault + billing-isolation policies | FMAPI in same workspace, same governance plane, same billing as data |
# MAGIC | Trace capture | Langfuse / Helicone / Arize Phoenix as separate services | `mlflow.openai.autolog()` — one line, same MLflow you already use for classic ML |
# MAGIC | Prompt versioning | Prompts in Git + custom loader + bespoke alias scheme | Prompt Registry with first-class `{{var}}` templates, immutable versions, alias API |
# MAGIC | Audit trail | Build it: structured logs → S3 → Athena → bespoke compliance reports | MLflow Traces are queryable via `search_traces`, viewable in UI, retained per experiment-retention policy |
# MAGIC | Cost attribution | Per-call token logging + custom cost calculator | Token counts captured automatically per trace; Databricks usage system tables roll up FMAPI cost by workspace + user |
# MAGIC
# MAGIC **How this composes in production**
# MAGIC
# MAGIC The Prompt Registry alias you set here (`@production`) is what every downstream module loads. When prompt engineering iterates in Module 9, bumping the alias to v2 propagates instantly to the RAG chain (M7) and the agent (M8) — without redeploying anything. The trace tree you build via `@mlflow.trace` becomes the production debugging tool: when a retention email goes wrong in production, search the traces for that customer_id and walk the spans from agent → tool → LLM call.
# MAGIC
# MAGIC **What's next — Module 7: RAG over support tickets**
# MAGIC
# MAGIC The Vector Search endpoint we kicked off in cell 2 is provisioning in the background. Module 7 will build a Delta Sync index on top of `support_tickets`, assemble a traced RAG chain, and answer churn-driver questions. Open `modules/07_rag_churn_insights/07_rag_churn_insights.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)
# MAGIC - [Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)
# MAGIC - [Foundation Model APIs — supported models](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models)
# MAGIC - [FMAPI rate limits](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits)
