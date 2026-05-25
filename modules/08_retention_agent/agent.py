"""Retention Outreach Agent — ResponsesAgent subclass (Models-from-Code entry point).

Logged via `mlflow.pyfunc.log_model(python_model="agent.py", ...)` from the driver
notebook `08_retention_agent.py`. The `mlflow.models.set_model(...)` call at the bottom
designates `RetentionAgent` as the model object MLflow loads at inference time.

Architecture
------------
- Inherits `mlflow.pyfunc.ResponsesAgent` (canonical 2026 Databricks agent flavor).
- Uses **OpenAI Agents SDK** (`from agents import Agent, Runner, function_tool`) for the
  inner tool-loop (per Q1 of PLAN.md).
- Exposes two tools that wire back to earlier workshop modules:
    1. `churn_score_tool(customer_id)` — POSTs to the Module 4 churn endpoint.
    2. `tickets_tool(customer_id, query)` — calls the Module 7 Vector Search index,
       filtered by customer.
- Customer features + endpoint/index identifiers are passed in via JSON **artifacts**
  (not env vars) so the agent works identically locally and on Model Serving.

Artifacts the driver notebook passes via `log_model(artifacts=...)`:
- `customer_features` — JSON mapping `customer_id → features dict`
- `agent_config` — JSON with `CHURN_ENDPOINT`, `VS_ENDPOINT`, `VS_INDEX`, `CHAT_MODEL`
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from typing import Any

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)


def _run_agent_in_fresh_loop(agent, user_msg):
    """Run an OpenAI Agents SDK ``Runner`` in a fresh thread + asyncio loop.

    ``Runner.run_sync`` raises ``RuntimeError`` when invoked from inside an
    already-running event loop, which is the default state in Databricks
    notebooks (and any Jupyter/IPython kernel). Running the async ``Runner.run``
    in a worker thread gives it its own loop and sidesteps the conflict.
    Works identically in notebooks, deployed Model Serving endpoints, and
    plain scripts.
    """
    from agents import Runner

    def _target():
        return asyncio.run(Runner.run(agent, user_msg))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_target).result()


def _build_agent_model():
    """Build an OpenAI Agents SDK model object backed by Databricks FMAPI Chat Completions.

    The Agents SDK defaults to the OpenAI Responses API. Databricks' FMAPI passthrough
    only supports the Responses API for OpenAI-native models (the GPT-5 family); Claude
    and other non-OpenAI models error with:
        BadRequestError: Responses API passthrough is not supported for model X

    Forcing ``OpenAIChatCompletionsModel`` routes calls through ``/chat/completions``
    instead, which Databricks supports for all FMAPI models including Claude.

    We use ``AsyncOpenAI`` (not ``OpenAI``) because the Agents SDK is async under the hood.
    """
    from openai import AsyncOpenAI

    # The adapter's import path has moved across openai-agents minor versions;
    # try both common locations.
    try:
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    except ImportError:
        from agents import OpenAIChatCompletionsModel  # type: ignore

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
    )
    return OpenAIChatCompletionsModel(
        model=_AgentConfig.chat_model,
        openai_client=client,
    )


class _AgentConfig:
    """Mutable container populated in `RetentionAgent.load_context`."""

    churn_endpoint: str = ""
    vs_endpoint: str = ""
    vs_index: str = ""
    chat_model: str = "databricks-claude-haiku-4-5"


def _configure_openai_for_databricks() -> None:
    """Point the OpenAI Agents SDK at the Databricks Foundation Model APIs.

    Sets OPENAI_BASE_URL / OPENAI_API_KEY env vars so the underlying OpenAI client
    (used internally by the OpenAI Agents SDK) reaches the Databricks FMAPI instead
    of api.openai.com.

    Auth-source priority:
        1. Direct env vars: DATABRICKS_HOST / DATABRICKS_WORKSPACE_URL +
           DATABRICKS_TOKEN / DATABRICKS_API_TOKEN. Driver notebooks set these
           explicitly; Databricks Model Serving usually injects them when the
           model is logged with `resources=[...]` declarations.
        2. Fallback to the Databricks SDK's WorkspaceClient auto-detection. Covers
           service-principal auth, workload identity, and any environment where
           the direct env vars haven't been populated.

    No-op if neither source yields creds (e.g., during a local-only test outside a
    Databricks environment).
    """
    host = (
        os.environ.get("DATABRICKS_HOST")
        or os.environ.get("DATABRICKS_WORKSPACE_URL")
    )
    token = (
        os.environ.get("DATABRICKS_TOKEN")
        or os.environ.get("DATABRICKS_API_TOKEN")
    )
    if not (host and token):
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            host = host or w.config.host
            token = token or w.config.token
        except Exception:
            pass

    if host and token:
        os.environ["OPENAI_BASE_URL"] = f"{host.rstrip('/')}/serving-endpoints"
        os.environ["OPENAI_API_KEY"] = token


def _call_churn_endpoint(features: dict[str, Any]) -> float:
    """POST to the Module 4 churn endpoint via the MLflow Deployments client."""
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")
    payload = {
        "dataframe_split": {
            "columns": list(features.keys()),
            "data": [list(features.values())],
        }
    }
    result = client.predict(endpoint=_AgentConfig.churn_endpoint, inputs=payload)
    preds = result.get("predictions", [])
    if not preds:
        return 0.0
    return float(preds[0])


def _query_vector_search(customer_id: str, query: str, k: int = 5) -> list[dict]:
    """Filter the VS index by customer_id and return the top-k support tickets."""
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    idx = vsc.get_index(endpoint_name=_AgentConfig.vs_endpoint, index_name=_AgentConfig.vs_index)
    results = idx.similarity_search(
        query_text=query,
        columns=["ticket_id", "customer_id", "category", "sentiment", "description"],
        num_results=k,
        filters={"customer_id": customer_id},
    )
    rows = results.get("result", {}).get("data_array", [])
    if not rows:
        return []
    cols = [c["name"] for c in results["manifest"]["columns"]]
    return [dict(zip(cols, row)) for row in rows]


def _extract_user_message(request: ResponsesAgentRequest) -> str:
    """Pull the most recent user message text from the Responses API input list."""
    for item in request.input:
        if isinstance(item, dict):
            item_dict = item
        elif hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        else:
            continue
        if item_dict.get("role") != "user":
            continue
        content = item_dict.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") in ("input_text", "text")
            )
    return ""


class RetentionAgent(ResponsesAgent):
    """Drafts a personalized retention email for a single bolttech customer."""

    def load_context(self, context):  # noqa: D401
        """Load the customer-features lookup + agent config artifacts, configure OpenAI."""
        # Customer features lookup
        features_artifact = context.artifacts.get("customer_features")
        if features_artifact and os.path.exists(features_artifact):
            with open(features_artifact) as fh:
                self.features_by_customer = json.load(fh)
        else:
            self.features_by_customer = {}

        # Agent config (endpoint/index names + chat model)
        config_artifact = context.artifacts.get("agent_config")
        if config_artifact and os.path.exists(config_artifact):
            with open(config_artifact) as fh:
                cfg = json.load(fh)
            _AgentConfig.churn_endpoint = cfg.get("CHURN_ENDPOINT", _AgentConfig.churn_endpoint)
            _AgentConfig.vs_endpoint = cfg.get("VS_ENDPOINT", _AgentConfig.vs_endpoint)
            _AgentConfig.vs_index = cfg.get("VS_INDEX", _AgentConfig.vs_index)
            _AgentConfig.chat_model = cfg.get("CHAT_MODEL", _AgentConfig.chat_model)

        _configure_openai_for_databricks()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # Defensive: ensure OpenAI client env vars are populated even if load_context()
        # was skipped (e.g. during MLflow's signature-inference predict at log_model time)
        # or if a Model Serving worker re-uses the process across cold starts.
        _configure_openai_for_databricks()

        user_msg = _extract_user_message(request)

        from agents import Agent, function_tool

        features_lookup = self.features_by_customer

        @function_tool
        def churn_score_tool(customer_id: str) -> str:
            """Return the churn risk score for the given customer ID."""
            features = features_lookup.get(customer_id)
            if not features:
                return f"No features available for customer {customer_id}."
            score = _call_churn_endpoint(features)
            return f"customer_id={customer_id} churn_score={score:.3f}"

        @function_tool
        def tickets_tool(customer_id: str, query: str = "churn cancel payment problem") -> str:
            """Retrieve the most relevant support tickets for the given customer."""
            tickets = _query_vector_search(customer_id, query)
            if not tickets:
                return f"No tickets found for customer {customer_id}."
            return json.dumps(
                [
                    {
                        "ticket_id": t["ticket_id"],
                        "category": t["category"],
                        "sentiment": t["sentiment"],
                        "description": t["description"][:300],
                    }
                    for t in tickets
                ]
            )

        agent = Agent(
            name="bolttech_retention",
            instructions=(
                "You are a customer-retention specialist at bolttech, a global insurtech.\n"
                "Given a customer ID, draft a short, personalized retention email. Your process:\n"
                "1. Call churn_score_tool with the customer ID to learn the risk.\n"
                "2. Call tickets_tool with the customer ID to retrieve their recent support tickets.\n"
                "3. Write a warm, professional retention email (<= 150 words) that:\n"
                "   - Acknowledges the specific issues you found in their tickets.\n"
                "   - Offers a concrete next step (e.g. a callback from CS, escalation, plan review).\n"
                "   - Mentions their plan tier if available in the tickets.\n"
                "   - Never promises a discount or refund amount.\n"
                "Use a tone that is empathetic but not overly casual. End with a clear sign-off."
            ),
            tools=[churn_score_tool, tickets_tool],
            # Explicit Chat Completions adapter — bypasses Databricks' Responses API
            # passthrough restriction that blocks Claude (and most non-OpenAI) FMAPI models.
            model=_build_agent_model(),
        )

        run_result = _run_agent_in_fresh_loop(agent, user_msg)
        text = str(run_result.final_output)

        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=text, id="msg_001")],
        )


mlflow.models.set_model(RetentionAgent())
