# Databricks notebook source
# MAGIC %md
# MAGIC # Module 07 — RAG: "Why are customers churning?"
# MAGIC ### Managed embeddings + Delta Sync — zero embed-pipeline plumbing
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > **TL;DR** — Point Vector Search at the `support_tickets` Delta table and the `databricks-gte-large-en` embedding endpoint. Databricks runs the embed + sync pipeline; you call `similarity_search` and wrap it in a `@mlflow.trace`-decorated retrieve → augment → generate chain. Multilingual ticket retrieval with zero ops overhead.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC We build a retrieval-augmented chain over the synthetic support tickets from Module 0. The retriever is a Databricks Vector Search **Delta Sync** index with **managed embeddings** (`databricks-gte-large-en`), and the generator is a chat call against `databricks-claude-haiku-4-5`. Every layer is traced.
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Create a Vector Search Delta Sync index with managed embeddings on the `support_tickets` table.
# MAGIC - Trigger and wait for the initial index sync.
# MAGIC - Run a `similarity_search` and inspect the structured results.
# MAGIC - Assemble a traced RAG chain (retrieve → augment → generate) and answer churn-driver questions.
# MAGIC
# MAGIC **Databricks features showcased**
# MAGIC
# MAGIC - **Databricks Vector Search Delta Sync index** — point the index at a Delta table column (`description`) and the embedding model (`databricks-gte-large-en`); Databricks runs the embedding + indexing pipeline. New rows in the source table flow into the index automatically on `pipeline_type='TRIGGERED'` or `'CONTINUOUS'`. No bespoke embedding worker, no separate vector DB, no cron job to keep things in sync.
# MAGIC - **Managed embeddings via FMAPI** — `embedding_model_endpoint_name='databricks-gte-large-en'` means Vector Search calls the Databricks-hosted embedding model directly; you never see a vector, never write embed code. (Compare to: spin up an embedding service, manage GPU quotas, handle retries, monitor latency, version the model — all gone.)
# MAGIC - **Delta + Change Data Feed as the source-of-truth pattern** — the same `support_tickets` Delta table is the source for analytics dashboards, RAG retrieval, training labels, and lineage tracking. One copy, governed by Unity Catalog.
# MAGIC - **Hybrid retrieval + structured filtering** — `similarity_search(filters={"customer_id": cust})` combines semantic search with structured predicates. The agent in M8 uses exactly this pattern to scope retrieval to one customer.
# MAGIC - **`@mlflow.trace(span_type=SpanType.RETRIEVER)`** — explicit retriever span makes the trace tree show the RAG structure (retrieve → parse → generate) instead of one flat LLM call. This is the diff between "the chain returned a bad answer" and "the retriever pulled the wrong tickets, so the LLM had no chance."
# MAGIC
# MAGIC **Why this matters for insurtech**
# MAGIC
# MAGIC bolttech's support tickets span ~14 languages and reference policy IDs, claim numbers, device serials, regional payment processors, and country-specific regulatory terms. A keyword-only search misses semantic matches ("payment kept bouncing" vs "transaction declined" vs "couldn't process my card"); a separate vector DB outside Unity Catalog breaks the lineage between tickets, policies, and CRM data. Vector Search with managed embeddings keeps the corpus in Delta (one copy), governed by UC (compliance team can audit who queried what), and queryable with both semantic + structured filters (find tickets *for this customer* matching *this churn pattern*).
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 0 and 6 have been run. (Module 6 kicked off the VS endpoint provisioning, which should be ready or close to ready by now.)
# MAGIC
# MAGIC **Expected runtime**: ~6-8 minutes (VS endpoint wait + index sync are the dominant costs).
# MAGIC
# MAGIC **Compute**: Serverless or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "openai>=1.50" \
# MAGIC   "databricks-vectorsearch>=0.50"
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
    TICKETS_TABLE,
    VS_ENDPOINT,
    VS_INDEX,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    EXPERIMENT_PATH,
    RAG_PROMPT_NAME,
    print_config,
)

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Wait for the VS endpoint to be `ONLINE`
# MAGIC
# MAGIC Module 6 kicked this off. If you ran Modules 6 and 7 back-to-back the endpoint should be most of the way there; otherwise this will block for a few minutes.

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)


def _endpoint_state(name: str) -> str:
    ep = vsc.get_endpoint(name=name)
    return ep.get("endpoint_status", {}).get("state", "UNKNOWN")


print(f"Waiting for VS endpoint {VS_ENDPOINT} to be ONLINE...")
deadline = time.time() + 600  # 10 min budget
while True:
    state = _endpoint_state(VS_ENDPOINT)
    if state == "ONLINE":
        print(f"  endpoint ONLINE")
        break
    if time.time() > deadline:
        raise TimeoutError(f"VS endpoint {VS_ENDPOINT} did not reach ONLINE in 10 min (last state: {state})")
    print(f"  state={state}, sleeping 15s...")
    time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Create the Delta Sync index (managed embeddings)
# MAGIC
# MAGIC The key choice here is **managed embeddings** — we point at the source Delta table's text column (`description`) and the embedding model endpoint (`databricks-gte-large-en`), and Vector Search handles the embed-sync-store pipeline automatically. The alternative (Direct Access) requires us to compute embeddings and upsert them ourselves.
# MAGIC
# MAGIC `pipeline_type="TRIGGERED"` is the right call for a workshop: cheaper than `CONTINUOUS`, syncs on-demand. We trigger a sync manually below.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search

# COMMAND ----------

# Idempotent: if the index already exists, skip creation
existing_indexes = vsc.list_indexes(name=VS_ENDPOINT).get("vector_indexes", [])
existing_names = {ix["name"] for ix in existing_indexes}

if VS_INDEX in existing_names:
    print(f"Index {VS_INDEX} already exists; skipping creation.")
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
else:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        source_table_name=TICKETS_TABLE,
        index_name=VS_INDEX,
        pipeline_type="TRIGGERED",
        primary_key="ticket_id",
        embedding_source_column="description",
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )
    print(f"Created Delta Sync index: {VS_INDEX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Wait for the initial sync to complete
# MAGIC
# MAGIC On 500 tickets this is fast (~1 minute). The index isn't queryable until at least the first sync finishes.

# COMMAND ----------

print(f"Waiting for index {VS_INDEX} to be ready / online...")
deadline = time.time() + 600
while True:
    desc = index.describe()
    status = desc.get("status", {})
    detailed_state = str(status.get("detailed_state", "")).upper()
    ready = bool(status.get("ready"))
    if ready or "ONLINE" in detailed_state:
        print(f"  index READY (detailed_state={detailed_state})")
        break
    if time.time() > deadline:
        raise TimeoutError(f"Index did not become ready in 10 min (last status: {status})")
    print(f"  status={status.get('detailed_state', 'INITIALIZING')}, sleeping 10s...")
    time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Test a similarity_search
# MAGIC
# MAGIC Verify retrieval works before we wire it into a chain. The query returns the top-k tickets along with cosine similarity scores.

# COMMAND ----------

raw_results = index.similarity_search(
    query_text="customer cancelling because of repeated payment failures",
    columns=["ticket_id", "customer_id", "category", "sentiment", "description"],
    num_results=5,
)
# `raw_results` shape: {"manifest": {"columns": [...]}, "result": {"data_array": [[..], [..]]}, "next_page_token": ...}
print(f"Returned {len(raw_results['result']['data_array'])} results.")
for row in raw_results["result"]["data_array"]:
    ticket_id, cust_id, cat, sent, desc = row[:5]
    score = row[-1]
    print(f"\n  [score={score:.3f}] {ticket_id} ({cat}/{sent}) cust={cust_id}")
    print(f"  {desc[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Register the RAG prompt template
# MAGIC
# MAGIC Distinct from Module 6's customer-specific summary prompt — this one is for open-ended "what's going on?" questions over the ticket corpus.

# COMMAND ----------

import mlflow

# UC-backed Prompt Registry — RAG_PROMPT_NAME is a 3-part `<catalog>.<schema>.<prompt>`
# identifier and requires the registry URI to be set to databricks-uc.
mlflow.set_registry_uri("databricks-uc")

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.openai.autolog()

RAG_PROMPT_TEMPLATE = """You are an insurtech analyst at bolttech, answering questions about customer churn signals from a corpus of support tickets.

QUESTION: {{question}}

Most relevant support tickets (highest cosine similarity first):
{{tickets}}

Instructions:
- Answer ONLY using the retrieved tickets. If they don't address the question, say so explicitly.
- Cite the ticket IDs you use in parentheses, e.g. (TIC_000123, TIC_000456).
- Be concrete — name patterns, not platitudes.
- Respond in <= 150 words.
"""

prompt = mlflow.genai.register_prompt(
    name=RAG_PROMPT_NAME,
    template=RAG_PROMPT_TEMPLATE,
    commit_message="Initial RAG QA prompt for churn ticket corpus",
    tags={"task": "rag_qa", "model_target": CHAT_MODEL},
)
mlflow.genai.set_prompt_alias(name=RAG_PROMPT_NAME, alias="production", version=prompt.version)
print(f"Registered {RAG_PROMPT_NAME} v{prompt.version} → @production")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7. Build the traced RAG chain
# MAGIC
# MAGIC Three layers: `retrieve` (a `RETRIEVER` span), `format_context` (a plain function call, also traced), and `generate` (LLM call, auto-traced by openai.autolog). Wrapped in a top-level `CHAIN` span.

# COMMAND ----------

from openai import OpenAI
from mlflow.entities import SpanType

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
client = OpenAI(api_key=ctx.apiToken().get(), base_url=f"{ctx.apiUrl().get()}/serving-endpoints")
loaded_prompt = mlflow.genai.load_prompt(f"prompts:/{RAG_PROMPT_NAME}@production")


@mlflow.trace(name="retrieve_tickets", span_type=SpanType.RETRIEVER)
def retrieve_tickets(question: str, k: int = 5) -> list[dict]:
    results = index.similarity_search(
        query_text=question,
        columns=["ticket_id", "customer_id", "category", "sentiment", "description"],
        num_results=k,
    )
    cols = [c["name"] for c in results["manifest"]["columns"]]
    return [dict(zip(cols, row)) for row in results["result"]["data_array"]]


@mlflow.trace(name="format_context", span_type=SpanType.PARSER)
def format_context(tickets: list[dict]) -> str:
    lines = []
    for t in tickets:
        lines.append(f"({t['ticket_id']}, {t['category']}/{t['sentiment']}, cust={t['customer_id']}): {t['description']}")
    return "\n".join(lines)


@mlflow.trace(name="churn_rag_qa", span_type=SpanType.CHAIN)
def churn_rag_qa(question: str, k: int = 5) -> dict:
    tickets = retrieve_tickets(question, k=k)
    context = format_context(tickets)
    filled = loaded_prompt.format(question=question, tickets=context)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": filled}],
        max_tokens=300,
    )
    return {
        "answer": resp.choices[0].message.content,
        "cited_ticket_ids": [t["ticket_id"] for t in tickets],
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 8. Run the chain on a few questions

# COMMAND ----------

for question in [
    "What are the most common churn drivers in our support tickets?",
    "Are there specific issues driving customers in Singapore to want to cancel?",
    "What technical problems are surfacing across multiple tickets?",
]:
    result = churn_rag_qa(question)
    print(f"\n=== Q: {question} ===")
    print(result["answer"])
    print(f"  Cited tickets: {result['cited_ticket_ids']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 9. View the trace tree
# MAGIC
# MAGIC Open the experiment's **Traces** tab to see the full hierarchy: `churn_rag_qa` → `retrieve_tickets` → `format_context` → the LLM call. This is exactly what you want when debugging RAG quality issues — you can see which tickets came back, what the augmented prompt looked like, and what the LLM did with it.

# COMMAND ----------

# Programmatic peek at the most recent trace. This cell is purely informational —
# the Traces UI tab is the canonical place to inspect traces. We wrap defensively
# because `search_traces`'s return-shape and `order_by` field names have churned
# across MLflow 3 minor versions.
try:
    _experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id
    # Drop `order_by` — its accepted field names ("start_time" vs "timestamp_ms" vs
    # "attributes.timestamp") vary by version. max_results=1 still returns one trace.
    recent = mlflow.search_traces(locations=[_experiment_id], max_results=1)
    if len(recent):
        row = recent.iloc[0]
        # Latency column name varies: `execution_time_ms` (legacy) vs
        # `execution_duration` (timedelta) vs absent entirely.
        if "execution_time_ms" in recent.columns:
            latency_str = f"{row['execution_time_ms']} ms"
        elif "execution_duration" in recent.columns and row["execution_duration"] is not None:
            td = row["execution_duration"]
            latency_str = (
                f"{td.total_seconds() * 1000:.0f} ms" if hasattr(td, "total_seconds") else f"{td} ms"
            )
        else:
            latency_str = "latency unavailable"
        print(f"Most recent trace: {row['trace_id']} ({latency_str})")
    else:
        print("No traces found in this experiment yet.")
except Exception as exc:
    print(f"(trace peek skipped: {type(exc).__name__}: {exc}) — view traces in the UI instead")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just built**
# MAGIC
# MAGIC - A Delta Sync Vector Search index over the support_tickets table with managed embeddings — no embed-and-upsert plumbing on your end.
# MAGIC - A traced RAG chain with clean `RETRIEVER` / `PARSER` / `CHAIN` spans surfaced in the MLflow Traces UI.
# MAGIC - A registered, aliased RAG prompt that Modules 8 and 9 will load by `@production`.
# MAGIC
# MAGIC **What you'd build without Databricks**
# MAGIC
# MAGIC | Concern | DIY stack | Databricks-native |
# MAGIC | --- | --- | --- |
# MAGIC | Embedding model hosting | Self-hosted SentenceTransformers + GPU autoscaling, or third-party API + key vault | FMAPI `databricks-gte-large-en` — one endpoint name, no infra to manage |
# MAGIC | Vector DB | Pinecone / Weaviate / Qdrant — separate account, separate billing, separate governance | Vector Search inside the same UC catalog as the source Delta table |
# MAGIC | Source-to-index sync | Custom worker reading from Kafka/CDC + retry logic + dead-letter queues | Delta Sync index — declarative; Databricks does the embed pipeline |
# MAGIC | Multi-language support | Manage one embedding model per language OR build a translation pipeline | `databricks-gte-large-en` is multilingual; one index covers SG/MY/ID/TH/JP/etc out of the box |
# MAGIC | Trace + debug | OpenTelemetry pipeline → Jaeger / Datadog — separate observability stack | `@mlflow.trace` decorators land in the same MLflow experiment UI as classic ML runs |
# MAGIC
# MAGIC **How this composes in production**
# MAGIC
# MAGIC The VS index updates automatically as new tickets land in the source Delta table (re-trigger `index.sync()` on a Job schedule, or flip `pipeline_type='CONTINUOUS'` for sub-minute freshness). The retrieve → augment → generate chain is portable: Module 8 wraps the same `index.similarity_search` call inside a `ResponsesAgent` tool, so the agent inherits all the trace/observability properties for free. The `@production` prompt alias means a prompt-engineering iteration in Module 9 propagates to both the standalone RAG chain AND the agent's retrieval tool without any code changes.
# MAGIC
# MAGIC **What's next — Module 8: Retention Outreach Agent**
# MAGIC
# MAGIC Module 8 wraps the Module 4 churn endpoint AND the index you just built into a `ResponsesAgent` that drafts personalized retention emails. We'll deploy that agent for real via `agents.deploy()`. Open `modules/08_retention_agent/08_retention_agent.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [Create & query Vector Search](https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search)
# MAGIC - [Delta Sync vs Direct Access](https://docs.databricks.com/aws/en/generative-ai/vector-search)
# MAGIC - [MLflow Tracing for RAG](https://mlflow.org/docs/latest/genai/tracing/)
# MAGIC - [Vector Search managed embedding models](https://docs.databricks.com/aws/en/generative-ai/vector-search#embedding-models)
