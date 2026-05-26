# Module 06 — Tracing & Prompt Registry Fundamentals

Make a chat call against the Databricks Foundation Model APIs (using the OpenAI-compatible client) with `mlflow.openai.autolog()` tracing enabled. Then register a prompt template in the MLflow Prompt Registry, set a `@production` alias, and load it by alias.

This module kicks off **Vector Search endpoint provisioning** in cell 1 (background) so Module 7 finds it ready — same background-provisioning pattern as Module 4's serving endpoint.

**Concepts covered**
- `mlflow.openai.autolog()` — auto-traces every `client.chat.completions.create(...)` call
- The OpenAI client against FMAPI: `openai.OpenAI(base_url=...)` with workspace token, or the `databricks-openai` helper
- The chosen chat model `databricks-claude-haiku-4-5` (rationale in `PLAN.md` §2.1)
- `@mlflow.trace(name=..., span_type=SpanType.LLM)` for manual tracing of wrapper functions
- Prompt Registry: `mlflow.genai.register_prompt(name, template)`, `{{var}}` double-curly template syntax
- `mlflow.genai.set_prompt_alias("name", alias="production", version=1)`
- Loading: `mlflow.genai.load_prompt("prompts:/<name>@production")`

**Prerequisites**
- Module 0 (the workshop catalog + schema must exist).

**Runtime target**: ~3 minutes.
**Compute**: Serverless or DBR 17.3 LTS ML.

**Notebook**: [`06_tracing_and_prompts.py`](./06_tracing_and_prompts.py)

---

> Status: scaffold stub.
