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

# UC catalog/schema. Default catalog is the generic workshop catalog `bolttech_workshop`.
# Override it WITHOUT editing code by setting the `WORKSHOP_CATALOG` env var — e.g. a customer
# pointing the workshop at an existing catalog in their own workspace. The per-user schema
# (`churn_<user>`) keeps participants from colliding on a shared workspace.
CATALOG: Final[str] = os.environ.get("WORKSHOP_CATALOG", "bolttech_workshop")
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
#
# UC-backed Prompt Registry (MLflow 3.12+) requires 3-part identifiers:
# `<catalog>.<schema>.<prompt_name>`. Each part must be alphanumeric+underscore.
# A flat name (no periods) causes the server to reject with
# "INVALID_PARAMETER_VALUE: name is not a valid name" because it can't parse
# the catalog/schema. Per-user isolation comes from FULL_SCHEMA already
# encoding USER_SLUG, so the prompt suffix doesn't need to repeat it.
SUMMARY_PROMPT_NAME: Final[str] = f"{FULL_SCHEMA}.churn_summary"
RAG_PROMPT_NAME: Final[str] = f"{FULL_SCHEMA}.churn_rag_qa"
EMAIL_PROMPT_NAME: Final[str] = f"{FULL_SCHEMA}.retention_email"

# Module 7 — Vector Search
VS_ENDPOINT: Final[str] = f"bolttech_vs_{USER_SLUG}"
VS_INDEX: Final[str] = f"{FULL_SCHEMA}.support_tickets_index"

# Module 8 — agent serving endpoint
AGENT_ENDPOINT: Final[str] = f"bolttech_agent_{USER_SLUG}"

# All modules — MLflow experiment path (per-user workspace path)
EXPERIMENT_PATH: Final[str] = f"/Users/{USER_EMAIL}/mlflow3_workshop"

# ---------------------------------------------------------------------------
# Observability — Unity Catalog trace storage + production monitoring (opt-in)
# ---------------------------------------------------------------------------
#
# The GenAI experiment's Overview dashboards (Usage / Quality / Tool calls) are powered
# by **Unity Catalog trace storage**, NOT the default control-plane trace store. To light
# them up, the experiment must be created UC-backed BEFORE its first trace (Module 0 does
# this) and a SQL warehouse must be configured for the monitoring queries.
#
# This is OPT-IN and OFF by default. Set MONITORING_WAREHOUSE_ID (via the `MONITORING_WAREHOUSE_ID`
# env var, or edit the default below) to a SQL warehouse ID to enable UC trace storage +
# the production-monitoring dashboard. Leave it empty to run the workshop anywhere with the
# default trace store — the Traces tab still works; only the aggregate Overview charts need
# UC storage.
#
# Prerequisites when enabled: MLflow 3.11+, a UC-enabled workspace, a SQL warehouse the
# runner can use, and the workspace's trace-storage preview features turned on.
# Ref: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
MONITORING_WAREHOUSE_ID: Final[str] = os.environ.get("MONITORING_WAREHOUSE_ID", "")
TRACE_TABLE_PREFIX: Final[str] = "mlflow_traces"  # UC Delta table prefix for OTel traces

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
        "MONITORING_WAREHOUSE_ID": MONITORING_WAREHOUSE_ID or "(unset — UC trace monitoring off)",
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
