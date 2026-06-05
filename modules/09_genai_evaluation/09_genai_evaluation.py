# Databricks notebook source
# MAGIC %md
# MAGIC # Module 09 — GenAI Evaluation with `mlflow.genai.evaluate`
# MAGIC ### LLM-as-judge regression testing — CI for prompts
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > **TL;DR** — Run `mlflow.genai.evaluate` on the Module 8 agent with a 25-example regression set scored by `Correctness`, `Safety`, and a custom `Guidelines("bolttech_voice", ...)` LLM-as-judge. Then register a v2 prompt, re-evaluate, and compare runs side-by-side in the MLflow UI. The pattern you'd block prompt-promotion on in CI.
# MAGIC
# MAGIC ---
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
# MAGIC **Databricks features showcased**
# MAGIC
# MAGIC - **`mlflow.genai.evaluate(data, predict_fn, scorers)`** — the unified MLflow 3 GenAI eval entry point. Replaces the old `mlflow.evaluate(model_type='databricks-agent')` pattern from MLflow 2.x. Same call works for a deployed endpoint, a locally-loaded agent, or a plain function — eval dataset → predict function → scorers → run.
# MAGIC - **Built-in LLM-as-judge scorers** — `Correctness()` (matches against `expected_facts`), `Safety()` (Databricks-managed PII / harm checks), `RetrievalGroundedness()`, `RelevanceToQuery()`, `Guidelines(name=..., guidelines=...)` (configurable natural-language rules). Each runs an LLM judgment per row and emits a structured Feedback record.
# MAGIC - **Custom `Guidelines` scorers** — codify domain-specific rules (bolttech brand voice, regulatory forbidden phrases, mandatory disclaimers) as natural-language constraints. The judge is itself an FMAPI call; no scoring infrastructure to build.
# MAGIC - **Prompt-iteration loop via the Prompt Registry** — register prompt v1 + v2 (Module 6 demonstrated the register/alias pattern), build a `predict_fn` per version, evaluate each. The two eval runs sit side-by-side in the MLflow UI for direct metric comparison.
# MAGIC - **`@mlflow.trace` per eval row** — wraps each `predict_fn` invocation in a named CHAIN span; the underlying agent / openai-autolog calls nest underneath, giving a tidy trace tree per row. Failed rows show up in the Traces tab with full context for debugging.
# MAGIC - **Eval datasets as Python modules** — `eval_dataset.py` is version-controlled in the repo. Diff a curated dataset like you'd diff code; reviewer can approve eval-set changes in PRs.
# MAGIC
# MAGIC **Why this matters for insurtech**
# MAGIC
# MAGIC The single biggest blocker to customer-facing GenAI at regulated insurers is *"how do you know it won't say something it shouldn't?"* `mlflow.genai.evaluate` with `Safety()` + a custom `Guidelines("bolttech_voice", ...)` scorer answers that with: every prompt change, every model swap, every retrieval-corpus update gets a 25-example regression check; results land in the MLflow UI; compliance can audit. The iteration loop matters even more — prompt engineering for production AI is a continuous activity, not a one-shot deploy. Without a structured eval loop, "let's tweak the prompt" becomes a slow chain of bug reports + manual spot-checks. With this pattern, it's a 5-minute CI cycle.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Modules 6, 7, 8 have been run.
# MAGIC
# MAGIC **Expected runtime**: ~5-6 minutes (25 examples × ~3 scorer judgments per example). The optional §9 production-monitoring section adds ~1 min to run, but its dashboard populates asynchronously (~15-20 min) — pre-stage it before a live session rather than running it inline.
# MAGIC
# MAGIC **Compute**: Serverless or DBR 17.3 LTS ML.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "mlflow[databricks]>=3.12,<4" \
# MAGIC   "openai>=1.50" \
# MAGIC   "openai-agents" \
# MAGIC   "databricks-vectorsearch>=0.50" \
# MAGIC   "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
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
    TICKETS_TABLE,
    N_EVAL_EXAMPLES,
    print_config,
)
from eval_dataset import build_examples, BOLTTECH_VOICE_GUIDELINES  # noqa: E402

print_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the eval set from agent-servable customers (derived, not hardcoded)
# MAGIC
# MAGIC We pick the eval customers at runtime: those with **support tickets in the Vector Search
# MAGIC corpus** (so the agent's retrieval tool can surface a *specific* issue), preferring
# MAGIC at-risk customers. This is the fix for the artificially-low `Correctness` score — hardcoding
# MAGIC sequential IDs meant the agent had no tickets/features for them and could only write generic
# MAGIC emails, which the judge correctly failed against the "specific issue from the customer's
# MAGIC tickets" expectation.

# COMMAND ----------

from pyspark.sql import functions as F

# Customers present in the ticket corpus (the VS index is synced from this table).
_with_tickets = spark.table(TICKETS_TABLE).select("customer_id").distinct()

# Prefer at-risk customers (predicted_churn desc) for a realistic retention scenario, then a
# stable customer_id tiebreak for determinism. Fall back to any ticketed customer if the
# batch_predictions table from Module 4 isn't available.
try:
    _preds = spark.table(f"{FULL_SCHEMA}.batch_predictions").select("customer_id", "predicted_churn")
    _candidates = _with_tickets.join(_preds, "customer_id", "left").orderBy(
        F.col("predicted_churn").desc_nulls_last(), F.col("customer_id")
    )
except Exception as exc:  # pragma: no cover — defensive
    print(f"  (batch_predictions unavailable: {exc}); selecting any ticketed customers")
    _candidates = _with_tickets.orderBy("customer_id")

eval_ids = [r["customer_id"] for r in _candidates.limit(N_EVAL_EXAMPLES).collect()]
EVAL_DATASET = build_examples(eval_ids)

print(f"Eval dataset: {len(EVAL_DATASET)} examples over agent-servable (ticketed) customers")
print(f"  customer IDs: {eval_ids}")
print(f"  sample example: {EVAL_DATASET[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
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
# MAGIC ---
# MAGIC ## 3. Define `predict_fn`
# MAGIC
# MAGIC `mlflow.genai.evaluate` calls `predict_fn(**inputs)` for each row. Our `inputs` dict has a single `message` key, so the function signature is `predict_fn(message: str) -> str`.

# COMMAND ----------

from mlflow.entities import SpanType

# `mlflow.openai.autolog()` (called in cell 5 below) auto-captures every
# `client.chat.completions.create(...)` invocation that happens inside the
# agent's tools. The @mlflow.trace decorator here adds an outer CHAIN span
# per eval row so the Traces tab shows one tidy trace per evaluation call,
# with the OpenAI / tool spans nested underneath.
@mlflow.trace(name="agent_predict_fn", span_type=SpanType.CHAIN)
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
# MAGIC ---
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
# MAGIC ---
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
# MAGIC ---
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
# MAGIC ---
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
from mlflow.entities import SpanType

# Re-assert tracing in this cell so (1) the Databricks UI's per-cell heuristic detects
# it and clears the "enable tracing" suggestion banner, and (2) we guarantee every
# OpenAI call below is captured as a trace even if a prior cell's autolog state was
# reset by some intervening operation.
mlflow.openai.autolog()

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

    @mlflow.trace(name=f"predict_fn_{prompt_alias}", span_type=SpanType.CHAIN)
    def predict_fn(message: str) -> str:
        cid = _extract_customer_id(message)
        filled = loaded.format(customer_id=cid)
        # The chat.completions.create call below is auto-traced by mlflow.openai.autolog()
        # and nests under this CHAIN span, producing a tidy retrieve→format→generate tree
        # in the Traces tab.
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
# MAGIC ---
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
# MAGIC ---
# MAGIC ## 9. (Optional) Production monitoring — populate the quality dashboard
# MAGIC
# MAGIC Everything above is **offline / on-demand** evaluation — you call `mlflow.genai.evaluate(...)` and it logs an Evaluation run. **Production monitoring** is the always-on counterpart: you *schedule* the same scorers to run automatically on a sample of the traces your app produces. Results then stream into the experiment's **monitoring dashboard** (the summary / "Overview"-style quality view) and attach as assessments on individual traces.
# MAGIC
# MAGIC Two-step lifecycle: **`scorer.register(name=...)`** then **`scorer.start(sampling_config=...)`**. After that, incoming traces are sampled and judged automatically. This is what fills the dashboard's **Quality** tab — *raw traces alone don't populate it; scheduled scorers do.*
# MAGIC
# MAGIC > **Prerequisite — Unity Catalog trace storage.** The Overview dashboards (Usage / Quality / Tool calls) read from **UC trace storage**, not the default trace store. Module 0 binds the experiment to UC storage and sets the monitoring SQL warehouse **when `MONITORING_WAREHOUSE_ID` is configured** (off by default — see `config/workshop_config.py`). If it wasn't set, the scorer assessments below still attach to each trace (visible in the Traces tab), but the aggregate Overview charts stay empty because there's no UC-backed trace data to aggregate.
# MAGIC
# MAGIC > **Instructor note — pre-stage / talk-track item, NOT a live-run step.** Scheduled scorers process traces **asynchronously (~15-20 min)** and add FMAPI judge cost, so the dashboard will not light up during the live session. Run this ~30 min *before* the workshop so the dashboard is populated when you demo it, then narrate it as "the same scorers from §4, now running continuously in production — the GenAI counterpart to Module 5's drift monitoring." Cleanup snippet is two cells down.
# MAGIC
# MAGIC Refs:
# MAGIC - https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring
# MAGIC - https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/production-quality-monitoring

# COMMAND ----------

# Ref: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring
import mlflow
from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig

# Scheduled scorers are scoped to the ACTIVE experiment.
mlflow.set_experiment(EXPERIMENT_PATH)

# Score 100% of sampled traffic so the workshop dashboard fills fast.
# In real production you'd sample (e.g. 0.1-0.7) to control LLM-judge cost.
MONITOR_SAMPLE_RATE = 1.0

# Idempotent: skip scorers already registered on this experiment so re-runs are safe.
try:
    from mlflow.genai.scorers import list_scorers, get_scorer
    _existing = {s.name for s in list_scorers()}
except Exception as exc:  # pragma: no cover — defensive across minor API revs
    print(f"(list_scorers unavailable: {exc}); proceeding without dedupe")
    _existing, get_scorer = set(), None


def _ensure_monitor(scorer_obj, register_name):
    """Register (if new) + schedule a scorer; safe to re-run."""
    if register_name in _existing and get_scorer is not None:
        print(f"  '{register_name}' already registered — reusing")
        sc = get_scorer(name=register_name)
    else:
        try:
            sc = scorer_obj.register(name=register_name)
            print(f"  registered '{register_name}'")
        except Exception as exc:
            print(f"  register '{register_name}' skipped ({exc}); attempting reuse")
            sc = get_scorer(name=register_name) if get_scorer else scorer_obj
    try:
        sc = sc.start(sampling_config=ScorerSamplingConfig(sample_rate=MONITOR_SAMPLE_RATE))
        print(f"  scheduled '{register_name}' @ sample_rate={MONITOR_SAMPLE_RATE}")
    except Exception as exc:
        print(f"  '{register_name}' already scheduled / start skipped: {exc}")
    return sc


# Schedule the same two judges we used offline above (Safety + the bolttech-voice Guidelines).
safety_monitor = _ensure_monitor(Safety(), "prod_safety")
voice_monitor = _ensure_monitor(
    Guidelines(name="bolttech_voice", guidelines=BOLTTECH_VOICE_GUIDELINES),
    "prod_bolttech_voice",
)

print("\nScheduled scorers now active on this experiment:")
try:
    for s in list_scorers():
        print(f"  - {s.name}: sample_rate={getattr(s, 'sample_rate', '?')}")
except Exception as exc:
    print(f"  (list unavailable: {exc})")

# COMMAND ----------

# Scheduled scorers judge INCOMING traces — emit a handful of fresh traces now to give the
# monitors something to sample. Assessments + dashboard charts surface asynchronously (~15-20 min).
mlflow.openai.autolog()

_sample_msgs = [ex["inputs"]["message"] for ex in EVAL_DATASET[:8]]
for i, _msg in enumerate(_sample_msgs):
    try:
        predict_v2(message=_msg)  # traced prompt-only fn from §7
        print(f"  emitted monitor trace {i + 1}/{len(_sample_msgs)}")
    except Exception as exc:
        print(f"  trace {i + 1} skipped: {exc}")

print(
    "\nMonitors scheduled. Open the experiment's monitoring dashboard in ~15-20 min to see "
    "quality trends; assessments will also attach to these traces in the Traces tab."
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Cleanup** — stop monitoring and remove the scheduled scorers when tearing down:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import delete_scorer
# MAGIC safety_monitor.stop(); voice_monitor.stop()
# MAGIC delete_scorer(name="prod_safety")
# MAGIC delete_scorer(name="prod_bolttech_voice")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Recap & handoff
# MAGIC
# MAGIC **What you just learned**
# MAGIC
# MAGIC - `mlflow.genai.evaluate(data, predict_fn, scorers)` is the unified MLflow 3 GenAI eval entry point.
# MAGIC - The eval dataset schema: each row has `inputs` (kwargs for `predict_fn`) + optional `expectations` (used by judges like `Correctness`).
# MAGIC - Three real scorers in action: `Correctness`, `Safety`, and a custom-instantiated `Guidelines`.
# MAGIC - The **prompt-iteration loop**: register a new prompt version → bump alias → re-evaluate → compare runs in the MLflow UI.
# MAGIC - **(Optional §9) Production monitoring**: `scorer.register(...).start(sampling_config=...)` schedules the same scorers to run continuously on live traces, populating the experiment's quality dashboard — the GenAI counterpart to Module 5's drift monitoring.
# MAGIC
# MAGIC **What you'd build without Databricks**
# MAGIC
# MAGIC | Concern | DIY stack | Databricks-native |
# MAGIC | --- | --- | --- |
# MAGIC | Eval framework | DeepEval / Ragas / promptfoo — separate library, different schema, different UI | `mlflow.genai.evaluate` — same MLflow you already use; eval runs sit alongside training runs |
# MAGIC | LLM-as-judge orchestration | Custom prompt + custom OpenAI/Claude call + JSON parse + retry | `Correctness()` / `Safety()` / `Guidelines(...)` — first-class scorer classes |
# MAGIC | Brand-voice / forbidden-phrase rules | Hand-roll regex checks OR build a custom judge | `Guidelines(name=..., guidelines="natural language rules")` — the judge IS the spec |
# MAGIC | Per-prompt-version comparison | Custom A/B framework + manual chart | Two `mlflow.start_run` blocks → MLflow UI chart-compare for free |
# MAGIC | Eval dataset versioning | CSVs in S3 + manual changelog | Python module in the repo → reviewable via PR like any other code |
# MAGIC | Per-row trace + failure debug | Print statements + grep through logs | Per-row trace in MLflow Traces tab, click any failed row to see full span tree |
# MAGIC
# MAGIC **How this composes in production**
# MAGIC
# MAGIC Wire this notebook into a Databricks Job that runs on every prompt-registry update (or on a schedule). The eval set grows over time — every customer-reported issue with a retention email becomes a new row in `eval_dataset.py`, with `expectations` codifying "the model should NOT have said X" or "should HAVE included Y". The Job blocks the prompt-alias promotion to `@production` until the new version's eval metrics meet a threshold (CI for prompts). Compliance can review eval results before any new prompt version reaches live customer-facing flows.
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
# MAGIC - [LLM-as-judge supported models](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/supported-models/)
