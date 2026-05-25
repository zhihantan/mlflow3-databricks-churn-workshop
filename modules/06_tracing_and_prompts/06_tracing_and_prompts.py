# Databricks notebook source
# MAGIC %md
# MAGIC # Module 06 — Tracing & Prompt Registry Fundamentals
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
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Module 0 has been run (we need the per-user catalog/schema to exist).
# MAGIC
# MAGIC **Expected runtime**: ~3 minutes.
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "openai>=1.50" \
# MAGIC   "databricks-vectorsearch>=0.50" \
# MAGIC   "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
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
# MAGIC ## 10. Verify everything by listing recent traces
# MAGIC
# MAGIC The experiment Traces tab is the UI view, but you can also query traces programmatically — useful for evaluation pipelines (Module 9).

# COMMAND ----------

# Ref: https://mlflow.org/docs/latest/genai/tracing/
recent_traces = mlflow.search_traces(experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id], max_results=10)
print(f"Recent traces in this experiment: {len(recent_traces)}")
display(recent_traces[["trace_id", "request", "response", "execution_time_ms"]] if len(recent_traces) else recent_traces)

# COMMAND ----------

# MAGIC %md
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
# MAGIC **What's next — Module 7: RAG over support tickets**
# MAGIC
# MAGIC The Vector Search endpoint we kicked off in cell 2 is provisioning in the background. Module 7 will build a Delta Sync index on top of `support_tickets`, assemble a traced RAG chain, and answer churn-driver questions. Open `modules/07_rag_churn_insights/07_rag_churn_insights.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)
# MAGIC - [Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)
# MAGIC - [Foundation Model APIs — supported models](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models)
