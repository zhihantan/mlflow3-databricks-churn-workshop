"""
workshop_config.py — Single source of truth for the MLflow 3 + Databricks Churn Workshop.

Every notebook in this repo imports identifiers from this module. The catalog/schema
strategy is **per-user** (locked decision Q2): each participant gets their own UC schema
and per-user serving endpoints so a shared workspace can host multiple participants
without collisions.

Usage in a Databricks notebook:

    import sys
    sys.path.append("/Workspace/Repos/<your-user>/mlflow3-databricks-churn-workshop")

    from config.workshop_config import (
        CATALOG, SCHEMA, FULL_SCHEMA,
        CHURN_MODEL_NAME, AGENT_MODEL_NAME,
        CHURN_ENDPOINT, AGENT_ENDPOINT,
        VS_ENDPOINT, VS_INDEX,
        CHAT_MODEL, EMBEDDING_MODEL,
        EXPERIMENT_PATH, print_config,
    )
    print_config()

The repo path-append shown above is the recommended pattern when this repo is cloned
via Databricks Git folders. If running outside Repos, add the repo root to `sys.path`
yourself.
"""
from __future__ import annotations

import os
import re
from typing import Final


# ---------------------------------------------------------------------------
# Per-user identity
# ---------------------------------------------------------------------------

def _current_user() -> str:
    """Return the active Databricks user's email.

    Strategy: try Spark `current_user()` first (works in any Databricks notebook),
    fall back to the `DATABRICKS_USER` env var (useful for local dev / CI), and
    finally to a sentinel default so importing this module never crashes outside
    of Databricks.
    """
    try:
        from pyspark.sql import SparkSession  # type: ignore

        spark = SparkSession.builder.getOrCreate()
        return spark.sql("SELECT current_user()").first()[0]
    except Exception:
        return os.environ.get("DATABRICKS_USER", "default_user@example.com")


def _sanitize_for_uc(s: str) -> str:
    """Lowercase + alphanum/underscore only + max 40 chars. Safe for UC schema /
    serving-endpoint / VS-endpoint names, all of which restrict character sets."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_").lower()
    return cleaned[:40] or "user"


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

USER_EMAIL: Final[str] = _current_user()
USER_SLUG: Final[str] = _sanitize_for_uc(USER_EMAIL.split("@")[0])

# UC catalog/schema — workshop locks catalog name to `bolttech_workshop` (Q4 default)
CATALOG: Final[str] = "bolttech_workshop"
SCHEMA: Final[str] = f"churn_{USER_SLUG}"
FULL_SCHEMA: Final[str] = f"{CATALOG}.{SCHEMA}"

# Module 0 — raw Delta tables
CUSTOMERS_TABLE: Final[str] = f"{FULL_SCHEMA}.customers"
POLICIES_TABLE: Final[str] = f"{FULL_SCHEMA}.policies"
CLAIMS_TABLE: Final[str] = f"{FULL_SCHEMA}.claims"
PAYMENTS_TABLE: Final[str] = f"{FULL_SCHEMA}.payments"
TICKETS_TABLE: Final[str] = f"{FULL_SCHEMA}.support_tickets"
SNAPSHOTS_TABLE: Final[str] = f"{FULL_SCHEMA}.customer_snapshots"

# Module 1 — UC feature table
FEATURE_TABLE: Final[str] = f"{FULL_SCHEMA}.customer_churn_features"

# Modules 2-4 — registered models (UC three-part name)
CHURN_MODEL_NAME: Final[str] = f"{FULL_SCHEMA}.bolttech_churn_model"
AGENT_MODEL_NAME: Final[str] = f"{FULL_SCHEMA}.bolttech_retention_agent"

# Module 4 — churn model serving endpoint (per-user)
CHURN_ENDPOINT: Final[str] = f"bolttech_churn_{USER_SLUG}"

# Module 5 — simulated inference table for monitoring
INFERENCE_TABLE: Final[str] = f"{FULL_SCHEMA}.churn_inferences_sim"

# Modules 6+ — FMAPI models (workspace-wide; not per-user)
CHAT_MODEL: Final[str] = "databricks-claude-haiku-4-5"
CHAT_MODEL_FALLBACK: Final[str] = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_MODEL: Final[str] = "databricks-gte-large-en"

# Module 6 — Prompt Registry
SUMMARY_PROMPT_NAME: Final[str] = f"churn_summary_{USER_SLUG}"
RAG_PROMPT_NAME: Final[str] = f"churn_rag_qa_{USER_SLUG}"
EMAIL_PROMPT_NAME: Final[str] = f"retention_email_{USER_SLUG}"

# Module 7 — Vector Search
VS_ENDPOINT: Final[str] = f"bolttech_vs_{USER_SLUG}"
VS_INDEX: Final[str] = f"{FULL_SCHEMA}.support_tickets_index"

# Module 8 — agent serving endpoint
AGENT_ENDPOINT: Final[str] = f"bolttech_agent_{USER_SLUG}"

# All modules — MLflow experiment path (per-user workspace path)
EXPERIMENT_PATH: Final[str] = f"/Users/{USER_EMAIL}/mlflow3_workshop"

# Synthetic data params (Module 0)
SYNTHETIC_SEED: Final[int] = 42
N_CUSTOMERS: Final[int] = 20_000
N_TICKETS: Final[int] = 500
SNAPSHOT_DATE_STR: Final[str] = "2026-04-01"  # the "as-of" date for the training snapshot

# Runtime tuning (Module 3)
N_OPTUNA_TRIALS: Final[int] = 15

# GenAI eval set size (Module 9)
N_EVAL_EXAMPLES: Final[int] = 25


def print_config() -> None:
    """Print every config value. Call at the top of any notebook for visual confirmation."""
    items: dict[str, object] = {
        "USER_EMAIL": USER_EMAIL,
        "USER_SLUG": USER_SLUG,
        "CATALOG": CATALOG,
        "SCHEMA": SCHEMA,
        "FULL_SCHEMA": FULL_SCHEMA,
        "CHURN_MODEL_NAME": CHURN_MODEL_NAME,
        "AGENT_MODEL_NAME": AGENT_MODEL_NAME,
        "CHURN_ENDPOINT": CHURN_ENDPOINT,
        "AGENT_ENDPOINT": AGENT_ENDPOINT,
        "VS_ENDPOINT": VS_ENDPOINT,
        "VS_INDEX": VS_INDEX,
        "CHAT_MODEL": CHAT_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "EXPERIMENT_PATH": EXPERIMENT_PATH,
        "SUMMARY_PROMPT_NAME": SUMMARY_PROMPT_NAME,
        "RAG_PROMPT_NAME": RAG_PROMPT_NAME,
        "EMAIL_PROMPT_NAME": EMAIL_PROMPT_NAME,
        "N_CUSTOMERS": N_CUSTOMERS,
        "N_TICKETS": N_TICKETS,
        "SNAPSHOT_DATE_STR": SNAPSHOT_DATE_STR,
        "N_OPTUNA_TRIALS": N_OPTUNA_TRIALS,
        "N_EVAL_EXAMPLES": N_EVAL_EXAMPLES,
    }
    width = max(len(k) for k in items)
    for k, v in items.items():
        print(f"  {k.ljust(width)} : {v}")


if __name__ == "__main__":
    print("Workshop config (per-user resolution):\n")
    print_config()
