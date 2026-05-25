# Module 08 — Retention Outreach Agent (`ResponsesAgent`)

Author a tool-using agent as a `mlflow.pyfunc.ResponsesAgent` subclass, log it via the Models-from-Code pattern with `resources=` declarations (for auto-auth passthrough), register it in UC, and call `agents.deploy()` to actually provision a Model Serving endpoint behind the Databricks Review App. While the endpoint provisions in the background, walk through the agent architecture and exercise the locally-loaded version.

This is the **most ambitious runtime module** — locked Q3 directs us to do a real `agents.deploy()` even though endpoint cold-start (~8-12 min) eats most of the module budget. The background-provisioning pattern from Module 4 is reused so participants see *all* of: log → register → deploy → live query in one module.

**Concepts covered**
- `mlflow.pyfunc.ResponsesAgent` — the canonical 2026 agent flavor (replaced `ChatAgent`)
- OpenAI Agents SDK as the inner-loop framework (locked Q1)
- Tool definition with `@function_tool` + type hints
- Two tools wiring back to earlier modules:
  - `get_customer_churn_score(customer_id)` → Module 4 serving endpoint
  - `retrieve_customer_tickets(customer_id, query)` → Module 7 VS index
- `mlflow.pyfunc.log_model(python_model="agent.py", resources=[...])` — Models-from-Code, auto-auth via `DatabricksServingEndpoint` + `DatabricksVectorSearchIndex` resources
- UC registration of the agent + `agents.deploy(uc_name, version)`
- Local invocation of the logged agent via `mlflow.pyfunc.load_model(f"models:/{model_id}")` for fast iteration
- The deployed endpoint with Review App URL + inference tables + real-time tracing

**Files in this folder**
- `08_retention_agent.py` — driver notebook (this is what participants run)
- `agent.py` — the `ResponsesAgent` subclass, logged via Models-from-Code

**Prerequisites**
- Modules 0, 4, 6, 7 have all been run. (Module 4's serving endpoint and Module 7's VS index must be ready.)

**Runtime target**: ~9 minutes (raised from 5 → 9 per locked Q3).
**Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

**Notebook**: [`08_retention_agent.py`](./08_retention_agent.py)

---

> Status: scaffold stub.
