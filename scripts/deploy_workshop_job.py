# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy & Run the Workshop e2e Job
# MAGIC ### One-click deployment of all 10 workshop modules — no CLI required
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > **TL;DR** — `Run All` this notebook to create the `MLFlow Workshop e2e job` in your Databricks Workflows + (optionally) trigger a run. Everything happens through the Python SDK using the notebook's own auth context. No env vars, no CLI install, no `databricks auth login`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## How it works
# MAGIC
# MAGIC ```
# MAGIC   ┌─────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
# MAGIC   │  Cell 2: pip + SDK  │ → │  Cell 3: build   │ → │  Cell 4: create or   │
# MAGIC   │  install            │   │  WorkspaceClient │   │  reset() the Job     │
# MAGIC   │                     │   │  + resolve paths │   │  (idempotent)        │
# MAGIC   └─────────────────────┘   └──────────────────┘   └──────────┬───────────┘
# MAGIC                                                               │
# MAGIC                                            ┌──────────────────┴───────────────────┐
# MAGIC                                            ▼                                      ▼
# MAGIC                              ┌─────────────────────────┐         ┌─────────────────────────┐
# MAGIC                              │  Workflows UI: click    │   OR    │  Cell 6: run_now() —    │
# MAGIC                              │  Run now on the new Job │         │  triggers from notebook │
# MAGIC                              └─────────────────────────┘         └─────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Why the SDK path (and not the CLI)?
# MAGIC
# MAGIC The `databricks` CLI is restricted in `%sh` cells on Serverless notebook compute (and most non-interactive contexts) — the platform itself recommends the Python SDK for programmatic API use from inside notebooks. This notebook uses `databricks.sdk.WorkspaceClient`, which:
# MAGIC
# MAGIC | Property | What it gives you |
# MAGIC | --- | --- |
# MAGIC | **Runs everywhere** | Serverless, classic clusters, and jobs alike — no environment-specific install. |
# MAGIC | **Zero auth setup** | Inherits the notebook's workspace credentials automatically. |
# MAGIC | **Identical Job output** | Produces a Job functionally indistinguishable from `databricks bundle deploy`. |
# MAGIC
# MAGIC > **Note** — the bundle files (`databricks.yml` + `resources/workshop_e2e_job.yml`) stay in the repo so users with local CLI access can still drive the deploy from a terminal (Quickstart C in the top-level README).
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Repo cloned to your workspace via **Repos** / **Git folders** (you're reading this inside the clone).
# MAGIC - Permission to create Jobs in the workspace — any standard developer role suffices.
# MAGIC
# MAGIC ## Expected runtime
# MAGIC
# MAGIC | Phase | Wall-clock |
# MAGIC | --- | --- |
# MAGIC | Deploy (cells 1–4) | **~5 seconds** |
# MAGIC | Optional Job run (cell 6 + e2e workshop) | **~40-60 minutes** |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 1 — Install the Databricks SDK
# MAGIC
# MAGIC The SDK is pre-installed on DBR ML LTS, but the Serverless ML base env can lag a release behind. An explicit pin + Python restart guarantees you're on a version that supports the Jobs API surface used below.
# MAGIC
# MAGIC **Expected output**
# MAGIC ```
# MAGIC Successfully installed databricks-sdk-0.40.0
# MAGIC Python interpreter will be restarted.
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 2 — Resolve the repo root and spin up the SDK client
# MAGIC
# MAGIC Two pieces of context the SDK needs:
# MAGIC
# MAGIC | What | Where it comes from |
# MAGIC | --- | --- |
# MAGIC | **Workspace repo path** | `dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath()` — strip two parents to get the repo root. |
# MAGIC | **Auth (host + token)** | `WorkspaceClient()` auto-discovers from the notebook runtime — no env vars to set, no profile to configure. |
# MAGIC
# MAGIC The cell sanity-checks the resolved path by confirming `databricks.yml` exists at the repo root, then prints the active user as proof of successful auth.

# COMMAND ----------

import os

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

# This notebook is at <repo>/scripts/deploy_workshop_job.py — strip two parents to
# get the repo root, then add the /Workspace prefix needed for absolute notebook paths.
_nb_path = ctx.notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(_nb_path))
REPO_ROOT = (
    _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
)
# Job notebook_path entries must be the WORKSPACE path WITHOUT the /Workspace prefix.
# We strip it back off here; REPO_ROOT (with prefix) is still used for os.path.exists checks.
REPO_ROOT_FOR_TASKS = REPO_ROOT[len("/Workspace"):] if REPO_ROOT.startswith("/Workspace") else REPO_ROOT

print(f"Workspace repo root (for sanity-checks): {REPO_ROOT}")
print(f"Notebook-task path prefix:               {REPO_ROOT_FOR_TASKS}")
print(f"databricks.yml present:                  {os.path.exists(os.path.join(REPO_ROOT, 'databricks.yml'))}")

if not os.path.exists(os.path.join(REPO_ROOT, "databricks.yml")):
    raise FileNotFoundError(
        f"Could not find databricks.yml at {REPO_ROOT}. "
        "Confirm this notebook is at <repo>/scripts/deploy_workshop_job.py inside the cloned repo."
    )

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print(f"\nWorkspaceClient host: {w.config.host}")
print(f"Current user:         {w.current_user.me().user_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 3 — Build the 10-task Job spec
# MAGIC
# MAGIC Mirrors `resources/workshop_e2e_job.yml` task-for-task. Each task is a `notebook_task` pointing at one workshop module; `depends_on` chains them into a single linear DAG.
# MAGIC
# MAGIC ### Task graph
# MAGIC
# MAGIC ```
# MAGIC  M1 ─► M2 ─► M3 ─► M4 ─► M5 ─► M6 ─► M7 ─► M8 ─► M9 ─► M10
# MAGIC  fe    LM    tune  reg   mon   tr    rag  agent eval  capstone
# MAGIC                          drift  +    + VS  ⏱30m
# MAGIC                                prom
# MAGIC ```
# MAGIC
# MAGIC ### Compute + timeout choices
# MAGIC
# MAGIC | Choice | Why |
# MAGIC | --- | --- |
# MAGIC | **No `new_cluster` / `job_cluster_key`** | Omitting compute config → tasks run on Serverless notebook compute (workspace policy mandates it). |
# MAGIC | **`timeout_seconds=1800` on M8 only** | `agents.deploy()` cold-start can run 8-12 min. 30 min headroom prevents premature task failure without letting a true hang block the chain forever. |
# MAGIC | **Linear chain via `depends_on`** | Mirrors how a participant would walk the workshop — each module reads state the previous one wrote to `workshop_state`. |

# COMMAND ----------

from databricks.sdk.service.jobs import Task, NotebookTask, TaskDependency

JOB_NAME = "MLFlow Workshop e2e job"

# (task_key, relative notebook path under <repo>/modules/)
_MODULES = [
    ("module_01_feature_engineering",   "01_feature_engineering/01_feature_engineering"),
    ("module_02_experiment_tracking",   "02_experiment_tracking/02_experiment_tracking"),
    ("module_03_tuning_and_eval",       "03_tuning_and_eval/03_tuning_and_eval"),
    ("module_04_registry_and_serving",  "04_registry_and_serving/04_registry_and_serving"),
    ("module_05_monitoring",            "05_monitoring/05_monitoring"),
    ("module_06_tracing_and_prompts",   "06_tracing_and_prompts/06_tracing_and_prompts"),
    ("module_07_rag_churn_insights",    "07_rag_churn_insights/07_rag_churn_insights"),
    ("module_08_retention_agent",       "08_retention_agent/08_retention_agent"),
    ("module_09_genai_evaluation",      "09_genai_evaluation/09_genai_evaluation"),
    ("module_10_capstone",              "10_capstone/10_capstone"),
]

tasks: list[Task] = []
_prev_key: str | None = None
for task_key, subpath in _MODULES:
    notebook_path = f"{REPO_ROOT_FOR_TASKS}/modules/{subpath}"
    task = Task(
        task_key=task_key,
        notebook_task=NotebookTask(notebook_path=notebook_path),
        depends_on=[TaskDependency(task_key=_prev_key)] if _prev_key else None,
        # M8's agents.deploy() cold-start can take 8-12 min. 30-min timeout = generous headroom.
        timeout_seconds=1800 if task_key == "module_08_retention_agent" else None,
    )
    tasks.append(task)
    _prev_key = task_key

print(f"Built {len(tasks)} chained tasks:")
for t in tasks:
    deps = [d.task_key for d in (t.depends_on or [])]
    print(f"  {t.task_key:35s} depends_on={deps}  notebook={t.notebook_task.notebook_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 4 — Create or update the Job (idempotent)
# MAGIC
# MAGIC Lookup-then-write pattern keeps re-runs safe:
# MAGIC
# MAGIC | Scenario | Action |
# MAGIC | --- | --- |
# MAGIC | **First run** | No job named `MLFlow Workshop e2e job` exists → `w.jobs.create(...)` provisions a new one and returns a fresh `job_id`. |
# MAGIC | **Re-run after `git pull`** | Job exists → `w.jobs.reset(job_id, new_settings=...)` overwrites task definitions in place. **Same `job_id`**, so existing run history is preserved. |
# MAGIC
# MAGIC > **Note** — `jobs.reset()` is a full-overwrite, not a merge. Tasks not present in the new settings are removed. This is what we want: the YAML / SDK definition is the source of truth.

# COMMAND ----------

from databricks.sdk.service.jobs import JobSettings

# Find an existing job with this name (if any)
existing = [j for j in w.jobs.list(name=JOB_NAME)]
print(f"Existing jobs named '{JOB_NAME}': {len(existing)}")

job_settings_kwargs = dict(
    name=JOB_NAME,
    tasks=tasks,
    max_concurrent_runs=1,
    tags={"purpose": "mlflow3-workshop-end-to-end-validation", "deployed_via": "sdk_notebook"},
)

if existing:
    job_id = existing[0].job_id
    print(f"Updating existing job_id={job_id}")
    w.jobs.reset(job_id=job_id, new_settings=JobSettings(**job_settings_kwargs))
else:
    created = w.jobs.create(**job_settings_kwargs)
    job_id = created.job_id
    print(f"Created new job_id={job_id}")

print(f"\nJob URL: {w.config.host}/jobs/{job_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Step 5 — (Optional) Trigger a run
# MAGIC
# MAGIC > **Heads up — long-running** — the full e2e workshop takes **~40-60 minutes** wall-clock. `run_now()` returns the `run_id` immediately and the run executes in the background; you don't need to keep this notebook attached.
# MAGIC
# MAGIC ### Two ways to start the run
# MAGIC
# MAGIC | From | How |
# MAGIC | --- | --- |
# MAGIC | **This notebook** | Execute the cell below — gives you a `run_id` + a direct URL to that specific run. |
# MAGIC | **Workflows UI** | Skip the cell below; open the Job URL printed by Step 4 and click **Run now**. |
# MAGIC
# MAGIC Pick whichever fits your demo flow.

# COMMAND ----------

run = w.jobs.run_now(job_id=job_id)
print(f"Triggered run_id={run.run_id}")
print(f"Run URL: {w.config.host}/jobs/{job_id}/runs/{run.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Tear-down
# MAGIC
# MAGIC The notebook deploys the Job; data resources are managed separately. The split is deliberate — you can iterate on Job task graphs without nuking your synthetic data, models, and endpoints.
# MAGIC
# MAGIC | What to clean up | How |
# MAGIC | --- | --- |
# MAGIC | **The Job itself** | Run this cell: `w.jobs.delete(job_id=job_id)` |
# MAGIC | **Workshop data resources** (catalog, schema, registered models, serving + VS endpoints) | Run [`scripts/reset_workshop.py`](./reset_workshop.py) — its own notebook with idempotent teardown steps. |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What you accomplished
# MAGIC
# MAGIC - Deployed an end-to-end validation Job using the Databricks Python SDK — zero CLI dependency, zero auth plumbing.
# MAGIC - Established a re-runnable, idempotent deployment pattern: `jobs.reset()` if exists, `jobs.create()` if not.
# MAGIC - (Optionally) triggered a run that chains all 10 workshop modules in dependency order on Serverless compute.
# MAGIC
# MAGIC Open the Job URL printed above to watch progress, or come back later — the run is detached from this notebook session.
