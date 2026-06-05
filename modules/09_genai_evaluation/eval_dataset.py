"""Evaluation-set BUILDER for the bolttech retention agent.

Customer IDs are **not hardcoded** — Module 9 derives them at runtime from customers the
agent can genuinely serve (those with support tickets in the Vector Search corpus, so the
agent's retrieval tool surfaces a *specific* issue), then calls `build_examples(...)`.

Why: an earlier version hardcoded sequential IDs (`CUST_000001, 000008, …`) on the
assumption they were high-risk customers with tickets. They weren't — 0/25 were in the
agent's served set and 0/25 had tickets — so the agent could only write generic emails and
the `Correctness` judge (which requires "a specific issue from the customer's tickets")
scored ~37%. Deriving real, servable customers makes Correctness measure response quality
instead of a data mismatch.

Each example follows the `mlflow.genai.evaluate` schema:

    {
        "inputs": {"message": "<user query>"},
        "expectations": {
            "expected_facts": [<list of strings the response should mention>],
            "guidelines": "<natural-language rules the response must follow>",
        }
    }

`predict_fn` takes a single `message` kwarg (the user input) and returns a string.
"""
from __future__ import annotations

# Query phrasings cycled across the eval customers so the set exercises varied inputs.
QUERY_PATTERNS = [
    "Draft a retention email for customer {cid}",
    "Customer {cid} appears at risk of churn — please write outreach to retain them",
    "Help me draft a personalized retention message for customer {cid}",
    "What outreach should we send to customer {cid}? They've shown cancellation signals",
    "Write a retention email for {cid} with specific recommendations from their support history",
]

BOLTTECH_VOICE_GUIDELINES = (
    "The response must: "
    "(1) mention at least one specific issue, theme, or detail from the customer's history; "
    "(2) propose a concrete next step (callback, escalation, plan review, dedicated CS contact); "
    "(3) NEVER promise a specific discount amount, refund amount, or premium reduction; "
    "(4) maintain a warm but professional tone, not overly casual or scripted; "
    "(5) be under 200 words."
)

EXPECTED_FACTS = [
    "specific issue or concern from the customer's tickets",
    "concrete next step or action",
    "professional tone appropriate for retention outreach",
]


def build_examples(customer_ids: list[str]) -> list[dict]:
    """Build the `mlflow.genai.evaluate` dataset for the given customer IDs.

    Pass IDs the agent can actually serve (customers with support tickets). Query phrasings
    are cycled deterministically so the set covers all five patterns.
    """
    examples: list[dict] = []
    for i, cid in enumerate(customer_ids):
        query = QUERY_PATTERNS[i % len(QUERY_PATTERNS)].format(cid=cid)
        examples.append(
            {
                "inputs": {"message": query},
                "expectations": {
                    "expected_facts": EXPECTED_FACTS,
                    "guidelines": BOLTTECH_VOICE_GUIDELINES,
                },
            }
        )
    return examples
