"""25-example evaluation dataset for the bolttech retention agent.

Each example follows the `mlflow.genai.evaluate` schema:

    {
        "inputs": {<kwargs passed to predict_fn>},
        "expectations": {
            "expected_facts": [<list of strings the response should mention>],
            "guidelines": "<natural-language rules the response must follow>",
        }
    }

The 25 examples vary across:
- Query phrasing (5 patterns)
- Customer IDs from the top-200 high-risk set (so the agent's tools have data to retrieve)
- Profile variations expected to surface different issues

`predict_fn` is expected to take a single `message` kwarg (the user input) and return a string.
"""
from __future__ import annotations

# Customer IDs chosen from the top-risk range used by Module 8's features lookup.
_QUERIES = [
    "Draft a retention email for customer {cid}",
    "Customer {cid} appears at risk of churn — please write outreach to retain them",
    "Help me draft a personalized retention message for customer {cid}",
    "What outreach should we send to customer {cid}? They've shown cancellation signals",
    "Write a retention email for {cid} with specific recommendations from their support history",
]

_BOLTTECH_VOICE_GUIDELINES = (
    "The response must: "
    "(1) mention at least one specific issue, theme, or detail from the customer's history; "
    "(2) propose a concrete next step (callback, escalation, plan review, dedicated CS contact); "
    "(3) NEVER promise a specific discount amount, refund amount, or premium reduction; "
    "(4) maintain a warm but professional tone, not overly casual or scripted; "
    "(5) be under 200 words."
)

_COMMON_EXPECTED_FACTS = [
    "specific issue or concern from the customer's tickets",
    "concrete next step or action",
    "professional tone appropriate for retention outreach",
]


def _build_examples() -> list[dict]:
    examples: list[dict] = []
    for i in range(25):
        cid = f"CUST_{(i * 7 + 1):06d}"  # Spread across top-risk pool; deterministic
        query = _QUERIES[i % len(_QUERIES)].format(cid=cid)
        examples.append(
            {
                "inputs": {"message": query},
                "expectations": {
                    "expected_facts": _COMMON_EXPECTED_FACTS,
                    "guidelines": _BOLTTECH_VOICE_GUIDELINES,
                },
            }
        )
    return examples


EVAL_DATASET: list[dict] = _build_examples()

BOLTTECH_VOICE_GUIDELINES = _BOLTTECH_VOICE_GUIDELINES
"""Exposed for the workshop notebook so the `Guidelines` scorer can reuse the same string."""
