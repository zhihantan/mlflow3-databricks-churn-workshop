# Databricks notebook source
# MAGIC %md
# MAGIC # Module 09 — GenAI Evaluation with `mlflow.genai.evaluate`
# MAGIC
# MAGIC Take the Module 8 agent and the 25-example eval dataset from `eval_dataset.py`, and run a complete LLM-as-judge evaluation: built-in `Correctness` + `Safety` scorers plus a custom `Guidelines` scorer that codifies the bolttech voice rules.
# MAGIC
# MAGIC We evaluate the **locally-loaded** agent (faster than the deployed endpoint, and independent of Module 8's deploy timing).
# MAGIC
# MAGIC **Learning objectives**
# MAGIC
# MAGIC By the end of this notebook you will:
# MAGIC
# MAGIC - Run `mlflow.genai.evaluate(...)` with a mix of built-in and custom scorers.
# MAGIC - Understand the eval dataset schema (`inputs` / `expectations` / `outputs` columns).
# MAGIC - See how the `predict_fn` contract works (kwargs unpacked from `inputs`).
# MAGIC - Demonstrate the **prompt iteration loop**: tweak a registered prompt, re-evaluate, compare runs in the MLflow UI.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 6, 7, 8 have been run.
# MAGIC
# MAGIC **Expected runtime**: ~5-6 minutes (25 examples × ~3 scorer judgments per example).
# MAGIC
# MAGIC **Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "openai>=1.50" \
# MAGIC   "openai-agents>=0.1" \
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
_nb_dir = os.path.dirname(_nb_path)
# modules/<module>/<notebook> → 3 dirnames to repo root
_repo_root_rel = os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
sys.path.append(_repo_root)

# Also add the current notebook directory so we can import eval_dataset.py
_module_dir = _nb_dir if _nb_dir.startswith("/Workspace") else "/Workspace" + _nb_dir
sys.path.append(_module_dir)

from config.workshop_config import (  # noqa: E402
    FULL_SCHEMA,
    EXPERIMENT_PATH,
    CHAT_MODEL,
    CHURN_ENDPOINT,
    VS_ENDPOINT,
    VS_INDEX,
    print_config,
)
from eval_dataset import EVAL_DATASET, BOLTTECH_VOICE_GUIDELINES  # noqa: E402

print_config()
print(f"\nEval dataset: {len(EVAL_DATASET)} examples")
print(f"Sample example: {EVAL_DATASET[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the local agent from Module 8

# COMMAND ----------

import mlflow

# UC-backed Prompt Registry — EMAIL_PROMPT_NAME is a 3-part `<catalog>.<schema>.<prompt>`
# identifier and requires the registry URI to be set to databricks-uc.
mlflow.set_registry_uri("databricks-uc")

# Set env vars the agent module needs at load time
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ.setdefault("DATABRICKS_HOST", ctx.apiUrl().get())
os.environ.setdefault("DATABRICKS_TOKEN", ctx.apiToken().get())
os.environ["CHURN_ENDPOINT"] = CHURN_ENDPOINT
os.environ["VS_ENDPOINT"] = VS_ENDPOINT
os.environ["VS_INDEX"] = VS_INDEX
os.environ["CHAT_MODEL"] = CHAT_MODEL

STATE_TABLE = f"{FULL_SCHEMA}.workshop_state"
state_rows = {r["key"]: r["value"] for r in spark.table(STATE_TABLE).select("key", "value").collect()}
agent_model_id = state_rows["agent_model_id"]
print(f"Loading agent model_id: {agent_model_id}")

local_agent = mlflow.pyfunc.load_model(f"models:/{agent_model_id}")
print("Agent loaded locally.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define `predict_fn`
# MAGIC
# MAGIC `mlflow.genai.evaluate` calls `predict_fn(**inputs)` for each row. Our `inputs` dict has a single `message` key, so the function signature is `predict_fn(message: str) -> str`.

# COMMAND ----------

def predict_fn(message: str) -> str:
    """Adapter: convert eval-row input to ResponsesAgent request and return the text output."""
    resp = local_agent.predict({"input": [{"role": "user", "content": message}]})
    parts: list[str] = []
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    parts.append(c.get("text", ""))
    return "\n".join(parts).strip()


# Quick smoke test
print("Smoke test on example 0:")
print(predict_fn(**EVAL_DATASET[0]["inputs"])[:400], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configure scorers
# MAGIC
# MAGIC Three scorers:
# MAGIC
# MAGIC 1. **`Correctness`** — built-in. Checks the response contains the `expected_facts` from `expectations`.
# MAGIC 2. **`Safety`** — built-in, Databricks-managed. Flags responses that contain unsafe content (promised illegal discounts, PII leakage, etc.).
# MAGIC 3. **`Guidelines("bolttech_voice", ...)`** — built-in scorer instantiated with our custom guideline string. LLM-judged.
# MAGIC
# MAGIC Refs:
# MAGIC - https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/
# MAGIC - https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/

# COMMAND ----------

from mlflow.genai.scorers import Correctness, Safety, Guidelines

scorers = [
    Correctness(),
    Safety(),
    Guidelines(name="bolttech_voice", guidelines=BOLTTECH_VOICE_GUIDELINES),
]
print(f"Scorers: {[s.name if hasattr(s, 'name') else type(s).__name__ for s in scorers]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run `mlflow.genai.evaluate`
# MAGIC
# MAGIC 25 examples × 3 scorers = 75 LLM-judge calls. With a default Databricks judge that's ~2-3 min wall-clock. Results are logged to a new MLflow run with per-row Feedback objects visible in the Traces tab.
# MAGIC
# MAGIC Ref: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/eval-examples

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.openai.autolog()

with mlflow.start_run(run_name="agent_eval_v1") as eval_run:
    results_v1 = mlflow.genai.evaluate(
        data=EVAL_DATASET,
        predict_fn=predict_fn,
        scorers=scorers,
    )

print(f"\nEval run: {eval_run.info.run_id}")
print(f"Metrics: {results_v1.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Inspect per-row results
# MAGIC
# MAGIC The `EvaluationResult` exposes a tables dict containing the per-row scores. Useful for inspecting failures and iterating.

# COMMAND ----------

# `results_v1.tables` typically contains "eval_results_table" with per-row outputs + scores
for table_name, table_df in results_v1.tables.items():
    print(f"\n=== Table: {table_name} ===")
    display(table_df.head(10) if hasattr(table_df, "head") else table_df[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Iteration demo: a v2 predict_fn with a tighter prompt
# MAGIC
# MAGIC We can't easily swap the *agent's* system instructions without re-logging the model, so to demonstrate the iteration loop concisely we evaluate a **simpler email-drafting function** loaded from a registered prompt. Bumping the prompt to v2 and re-evaluating produces a parallel MLflow run we can compare against the agent's run.

# COMMAND ----------

from config.workshop_config import EMAIL_PROMPT_NAME

EMAIL_PROMPT_V1 = """Draft a retention email for bolttech customer {{customer_id}}.

Constraints:
- Mention at least one specific issue or theme from their account history if known; otherwise reference generic insurtech concerns (claims latency, payment friction, plan fit).
- Propose a concrete next step (callback, escalation, plan review).
- Do not promise specific discounts or refunds.
- Keep the email under 150 words.
- Warm but professional tone.

Respond with just the email text — no preamble.
"""

EMAIL_PROMPT_V2 = """Draft a personalized retention email for bolttech customer {{customer_id}}.

REQUIRED elements:
1. Open with an acknowledgement — name a specific frustration the customer has likely experienced (recent payment failures, claim delays, plan-fit concerns).
2. Single concrete next step — a callback within 48h from a named human (Sarah from Customer Success), an account-tier review, OR a dedicated escalation contact.
3. Sign-off from "The bolttech retention team" — warm but professional.

FORBIDDEN:
- Any specific monetary discount, percentage off, or refund amount.
- Generic platitudes ("we value you", "thanks for being a customer") without specifics.
- Phrases longer than 25 words.

Response: just the email body. Under 150 words.
"""

p1 = mlflow.genai.register_prompt(
    name=EMAIL_PROMPT_NAME,
    template=EMAIL_PROMPT_V1,
    commit_message="V1 email-drafting prompt for iteration demo",
)
p2 = mlflow.genai.register_prompt(
    name=EMAIL_PROMPT_NAME,
    template=EMAIL_PROMPT_V2,
    commit_message="V2 — stricter structure + forbidden patterns",
)
mlflow.genai.set_prompt_alias(name=EMAIL_PROMPT_NAME, alias="v1", version=p1.version)
mlflow.genai.set_prompt_alias(name=EMAIL_PROMPT_NAME, alias="v2", version=p2.version)
mlflow.genai.set_prompt_alias(name=EMAIL_PROMPT_NAME, alias="production", version=p2.version)
print(f"Registered prompt versions: v1={p1.version} v2={p2.version} (production → v2)")

# COMMAND ----------

# MAGIC %md
# MAGIC Build a simple prompt-only `predict_fn` per version and evaluate each.

# COMMAND ----------

import re
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)


def _extract_customer_id(message: str) -> str:
    """Parse 'CUST_000NNN' out of the eval prompt."""
    match = re.search(r"CUST_\d{6}", message)
    return match.group(0) if match else "CUST_000001"


def _make_predict_fn(prompt_alias: str):
    loaded = mlflow.genai.load_prompt(f"prompts:/{EMAIL_PROMPT_NAME}@{prompt_alias}")

    def predict_fn(message: str) -> str:
        cid = _extract_customer_id(message)
        filled = loaded.format(customer_id=cid)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": filled}],
            max_tokens=300,
        )
        return resp.choices[0].message.content
    return predict_fn


predict_v1 = _make_predict_fn("v1")
predict_v2 = _make_predict_fn("v2")

with mlflow.start_run(run_name="prompt_only_eval_v1"):
    results_prompt_v1 = mlflow.genai.evaluate(
        data=EVAL_DATASET,
        predict_fn=predict_v1,
        scorers=scorers,
    )

with mlflow.start_run(run_name="prompt_only_eval_v2"):
    results_prompt_v2 = mlflow.genai.evaluate(
        data=EVAL_DATASET,
        predict_fn=predict_v2,
        scorers=scorers,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Compare v1 vs v2 metrics

# COMMAND ----------

import pandas as pd

comparison = pd.DataFrame(
    [
        {"variant": "agent_eval_v1", **{k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in results_v1.metrics.items()}},
        {"variant": "prompt_only_v1", **{k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in results_prompt_v1.metrics.items()}},
        {"variant": "prompt_only_v2", **{k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in results_prompt_v2.metrics.items()}},
    ]
)
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just learned**
# MAGIC
# MAGIC - `mlflow.genai.evaluate(data, predict_fn, scorers)` is the unified MLflow 3 GenAI eval entry point.
# MAGIC - The eval dataset schema: each row has `inputs` (kwargs for `predict_fn`) + optional `expectations` (used by judges like `Correctness`).
# MAGIC - Three real scorers in action: `Correctness`, `Safety`, and a custom-instantiated `Guidelines`.
# MAGIC - The **prompt-iteration loop**: register a new prompt version → bump alias → re-evaluate → compare runs in the MLflow UI.
# MAGIC
# MAGIC **What's next — Module 10: Capstone**
# MAGIC
# MAGIC Module 10 stitches everything together: batch-score 100 customers via Module 4's endpoint, rank top-10, invoke the deployed Module 8 agent on each, display the drafted retention emails. Open `modules/10_capstone/10_capstone.py`.
# MAGIC
# MAGIC **Go deeper**
# MAGIC - [`mlflow.genai.evaluate`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html)
# MAGIC - [Predefined judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/)
# MAGIC - [Custom scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/)
# MAGIC - [Agent eval migration notes (2.x → 3.x)](https://docs.databricks.com/aws/en/mlflow3/genai/agent-eval-migration)
