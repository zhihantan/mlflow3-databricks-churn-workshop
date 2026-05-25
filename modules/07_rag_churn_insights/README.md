# Module 07 — RAG: "Why are customers churning?"

Build a Delta Sync Vector Search index over the synthetic support tickets with managed embeddings (`databricks-gte-large-en`), then assemble a traced RAG chain that retrieves the most relevant tickets for a customer and answers churn-driver questions using the registered prompt from Module 6.

**Concepts covered**
- `VectorSearchClient` — endpoint + index lifecycle
- `create_delta_sync_index(..., pipeline_type="TRIGGERED", embedding_model_endpoint_name="databricks-gte-large-en")` — managed embeddings, low ops overhead
- Why **Delta Sync** beats Direct Access for a small synthetic corpus on a workshop budget
- `similarity_search(query_text=..., columns=[...], num_results=5)`
- Composing a RAG chain as a `@mlflow.trace`-decorated Python function — retrieve → augment → LLM call
- How retrieval traces show up in the MLflow Traces UI

**Prerequisites**
- Modules 0, 6 have been run (Module 6 already kicked off the VS endpoint provisioning).

**Runtime target**: ~8 minutes (VS endpoint pre-provisioned by Module 6; index sync on ~500 rows is fast).
**Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

**Notebook**: [`07_rag_churn_insights.py`](./07_rag_churn_insights.py)

---

> Status: scaffold stub.
