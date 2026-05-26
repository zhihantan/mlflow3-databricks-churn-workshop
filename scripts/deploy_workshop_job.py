# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy & Run the Workshop e2e Job — one-click from a Databricks notebook
# MAGIC
# MAGIC Run this notebook AFTER cloning the workshop repo into your Databricks workspace via
# MAGIC **Repos** / **Git folders**. It creates the same `MLFlow Workshop e2e job` that the
# MAGIC bundle (`databricks.yml` + `resources/workshop_e2e_job.yml`) defines, but does it
# MAGIC via the **Databricks Python SDK** instead of the `databricks bundle` CLI.
# MAGIC
# MAGIC **Why the SDK path?**
# MAGIC
# MAGIC The `databricks` CLI is restricted in `%sh` cells on Serverless notebook compute (and
# MAGIC most non-interactive job contexts) — the platform itself recommends the Python SDK
# MAGIC for programmatic API use from inside notebooks. This rewrite uses
# MAGIC `databricks.sdk.WorkspaceClient` which:
# MAGIC
# MAGIC - works everywhere a notebook runs (Serverless ✓, classic clusters ✓, jobs ✓);
# MAGIC - inherits the notebook's auth context automatically — no env vars, no CLI install;
# MAGIC - produces a Job that's functionally identical to what `bundle deploy` would create.
# MAGIC
# MAGIC The bundle files (`databricks.yml` + `resources/workshop_e2e_job.yml`) stay in the
# MAGIC repo for users who prefer the CLI workflow from their own terminal (Quickstart C in
# MAGIC the top-level README).
# MAGIC
# MAGIC **What this notebook does**
# MAGIC
# MAGIC 1. Resolves the workspace path of the cloned repo so notebook tasks point at the
# MAGIC    user's actual Git folder paths.
# MAGIC 2. Builds a 10-task Job spec mirroring `resources/workshop_e2e_job.yml`.
# MAGIC 3. **Idempotently** creates or updates the Job (looks up by name; updates if it
# MAGIC    already exists). Safe to re-run after pulling repo changes.
# MAGIC 4. Prints the Job URL so you can monitor in the Workflows UI.
# MAGIC 5. Optional final cell: triggers a run via `w.jobs.run_now(...)`.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Repo cloned to your workspace via Git folders (you're reading this inside the clone).
# MAGIC - Permission to create Jobs in the workspace (any standard developer role).
# MAGIC
# MAGIC **Expected runtime**
# MAGIC
# MAGIC - Deploy: ~5 seconds.
# MAGIC - Optional Job run (full e2e workshop): ~40-60 minutes if you execute the final cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ensure the Databricks SDK is installed
# MAGIC
# MAGIC Pre-installed on DBR ML LTS, but the Serverless ML base env can lag — explicit
# MAGIC install + restart guarantees a recent version.

# COMMAND ----------

# MAGIC %pip install --quiet "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Resolve the repo root + spin up the SDK client
# MAGIC
# MAGIC The notebook task definitions will reference the user's actual Git-folder paths in
# MAGIC the workspace, computed from this notebook's own `notebookPath()`. No env var
# MAGIC plumbing needed — `WorkspaceClient()` auto-discovers auth from the notebook context.

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
# MAGIC ## 3. Build the Job spec (mirrors `resources/workshop_e2e_job.yml`)
# MAGIC
# MAGIC 10 chained `notebook_task` entries, each pointing at the user's workspace path for
# MAGIC that module. No cluster spec → Serverless. `module_08_retention_agent` gets a 30-min
# MAGIC timeout to absorb the `agents.deploy()` cold-start.

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
# MAGIC ## 4. Create-or-update the Job (idempotent)
# MAGIC
# MAGIC Looks up by name first. If a job named `MLFlow Workshop e2e job` already exists
# MAGIC (from a previous run of this notebook), updates its settings in place via
# MAGIC `jobs.reset(...)`. Otherwise creates a fresh one via `jobs.create(...)`.
# MAGIC
# MAGIC Either way, the same `job_id` is reused on subsequent runs — re-running this notebook
# MAGIC won't accumulate duplicates.

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
# MAGIC ## 5. (Optional) Trigger a run from this notebook
# MAGIC
# MAGIC `w.jobs.run_now(...)` returns immediately with a `run_id` — the run executes in the
# MAGIC background. The full e2e takes ~40-60 min; monitor in the Workflows UI via the URL
# MAGIC printed above (or the URL printed by this cell).
# MAGIC
# MAGIC Skip this cell if you'd rather trigger from the Workflows UI ("Run now" button).

# COMMAND ----------

run = w.jobs.run_now(job_id=job_id)
print(f"Triggered run_id={run.run_id}")
print(f"Run URL: {w.config.host}/jobs/{job_id}/runs/{run.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tear-down
# MAGIC
# MAGIC To delete the Job entirely:
# MAGIC
# MAGIC ```python
# MAGIC w.jobs.delete(job_id=job_id)
# MAGIC ```
# MAGIC
# MAGIC To clean up the workshop's data resources (catalog, schema, registered models,
# MAGIC serving endpoints, VS endpoints + indexes), run [`scripts/reset_workshop.py`](./reset_workshop.py)
# MAGIC separately. That's a deliberate split — deleting the Job doesn't touch the data.
