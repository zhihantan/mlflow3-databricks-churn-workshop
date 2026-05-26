# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy & Run the Workshop e2e Job — one-click from a Databricks notebook
# MAGIC
# MAGIC Run this notebook AFTER cloning the workshop repo into your Databricks workspace via
# MAGIC **Repos** / **Git folders**. It deploys the Databricks Asset Bundle defined in
# MAGIC `databricks.yml` + `resources/workshop_e2e_job.yml` and (optionally) triggers a run
# MAGIC of the resulting `MLFlow Workshop e2e job`.
# MAGIC
# MAGIC **What this notebook does**
# MAGIC
# MAGIC 1. Resolves the workspace path of the cloned repo (where `databricks.yml` lives).
# MAGIC 2. Exports `DATABRICKS_HOST` + `DATABRICKS_TOKEN` from the notebook's own auth context
# MAGIC    so the Databricks CLI inherits valid credentials.
# MAGIC 3. Installs the **Databricks CLI** (the v0.205+ Go binary, which supports bundles) if
# MAGIC    it isn't already on PATH.
# MAGIC 4. Runs `databricks bundle validate` → `databricks bundle deploy --target dev` →
# MAGIC    `databricks bundle summary` to surface the deployed Job URL.
# MAGIC 5. (Optional last cell) Kicks off `databricks bundle run workshop_e2e --no-wait` so
# MAGIC    the chained 10-task Job starts immediately.
# MAGIC
# MAGIC **Why a notebook (vs running the CLI locally)**
# MAGIC
# MAGIC - Removes the "install + configure the CLI on the user's laptop" step from the
# MAGIC   getting-started path. Everything happens inside the workspace the user already has.
# MAGIC - The notebook context already has authenticated workspace creds; we just re-export
# MAGIC   them as env vars so the CLI sees them.
# MAGIC - One-click for customer demos: clone → open this notebook → Run All.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Repo cloned to your workspace via Git folders (you're presumably reading this
# MAGIC   inside that clone already).
# MAGIC - Permission to create Jobs in the workspace (any standard developer role).
# MAGIC - Outbound HTTPS access from the notebook host (needed for the one-time CLI install).
# MAGIC
# MAGIC **Expected runtime**
# MAGIC
# MAGIC - Deploy: ~30-60 seconds.
# MAGIC - Optional bundle run (full e2e workshop): ~40-60 minutes if you uncomment the final
# MAGIC   cell. Otherwise the cell prints the Job URL and you trigger from the Workflows UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resolve the repo root and set up CLI auth env vars
# MAGIC
# MAGIC We export `DATABRICKS_HOST` / `DATABRICKS_TOKEN` from the notebook's `dbutils` context.
# MAGIC The Databricks CLI auto-discovers these and uses them as the active profile — no
# MAGIC `databricks auth login` step required from within the notebook.

# COMMAND ----------

import os

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

# Export auth creds so the `%sh databricks ...` cells below can authenticate.
os.environ["DATABRICKS_HOST"] = ctx.apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()

# Resolve the workspace path of the repo root (where databricks.yml lives).
# This notebook is at: <repo>/scripts/deploy_workshop_job.py
# So `_repo_root_rel` strips two parents to get back to <repo>.
_nb_path = ctx.notebookPath().get()
_repo_root_rel = os.path.dirname(os.path.dirname(_nb_path))
_repo_root = _repo_root_rel if _repo_root_rel.startswith("/Workspace") else "/Workspace" + _repo_root_rel
os.environ["REPO_ROOT"] = _repo_root  # consumed by the %sh cells below

# Sanity checks
print(f"DATABRICKS_HOST: {os.environ['DATABRICKS_HOST']}")
print(f"REPO_ROOT:       {os.environ['REPO_ROOT']}")
print(f"databricks.yml present: {os.path.exists(os.path.join(_repo_root, 'databricks.yml'))}")
print(f"workshop_e2e_job.yml present: {os.path.exists(os.path.join(_repo_root, 'resources', 'workshop_e2e_job.yml'))}")

if not os.path.exists(os.path.join(_repo_root, "databricks.yml")):
    raise FileNotFoundError(
        f"Could not find databricks.yml at {_repo_root}. "
        "Confirm this notebook is at <repo>/scripts/deploy_workshop_job.py inside the cloned repo."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install the Databricks CLI (only if missing)
# MAGIC
# MAGIC The legacy `databricks-cli` Python package on PyPI does **not** support bundles. We
# MAGIC need the modern Go CLI (v0.205+). The setup-cli installer drops it at
# MAGIC `/usr/local/bin/databricks` if you have sudo, or `~/.databricks/bin/databricks`
# MAGIC otherwise — we add both to `PATH` defensively.

# COMMAND ----------

# MAGIC %sh
# MAGIC # Add common install locations to PATH for this %sh session
# MAGIC export PATH="$PATH:/usr/local/bin:$HOME/.databricks/bin:$HOME/.local/bin"
# MAGIC
# MAGIC if command -v databricks >/dev/null 2>&1; then
# MAGIC   echo "Databricks CLI already on PATH:"
# MAGIC   databricks --version
# MAGIC else
# MAGIC   echo "Databricks CLI not found — installing via the official setup-cli script..."
# MAGIC   curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
# MAGIC   export PATH="$PATH:/usr/local/bin:$HOME/.databricks/bin:$HOME/.local/bin"
# MAGIC   databricks --version
# MAGIC fi

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validate the bundle
# MAGIC
# MAGIC Catches typos in `databricks.yml` / `resources/*.yml` before any resources are
# MAGIC actually created in the workspace.

# COMMAND ----------

# MAGIC %sh
# MAGIC export PATH="$PATH:/usr/local/bin:$HOME/.databricks/bin:$HOME/.local/bin"
# MAGIC cd "$REPO_ROOT" && databricks bundle validate --target dev

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Deploy the bundle
# MAGIC
# MAGIC This is the part that actually creates Workspace resources:
# MAGIC
# MAGIC - Syncs all the notebooks + config into `${workspace.root_path}/files/...` so the
# MAGIC   Job's notebook tasks have a stable workspace path to reference.
# MAGIC - Creates the **`[dev <your-user>] MLFlow Workshop e2e job`** Job in Workflows.
# MAGIC
# MAGIC Re-running this cell after pulling repo changes updates the deployed Job in place.

# COMMAND ----------

# MAGIC %sh
# MAGIC export PATH="$PATH:/usr/local/bin:$HOME/.databricks/bin:$HOME/.local/bin"
# MAGIC cd "$REPO_ROOT" && databricks bundle deploy --target dev

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Show the deployed Job URL
# MAGIC
# MAGIC `bundle summary` prints a structured report of every resource the bundle owns,
# MAGIC including the URL of the `workshop_e2e` Job. Click that URL to monitor runs in the
# MAGIC Workflows UI.

# COMMAND ----------

# MAGIC %sh
# MAGIC export PATH="$PATH:/usr/local/bin:$HOME/.databricks/bin:$HOME/.local/bin"
# MAGIC cd "$REPO_ROOT" && databricks bundle summary --target dev

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. (Optional) Trigger a run from this notebook
# MAGIC
# MAGIC Uncomment + execute the cell below to kick off a Job run programmatically. We pass
# MAGIC `--no-wait` so the notebook doesn't block for the full ~40-60 min e2e duration;
# MAGIC monitor in the Workflows UI via the URL from cell 5.
# MAGIC
# MAGIC Alternative: open the Job URL in the UI and click **Run now** there.

# COMMAND ----------

# MAGIC %sh
# MAGIC export PATH="$PATH:/usr/local/bin:$HOME/.databricks/bin:$HOME/.local/bin"
# MAGIC cd "$REPO_ROOT" && databricks bundle run workshop_e2e --target dev --no-wait

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tear-down
# MAGIC
# MAGIC When you're done with the deployed Job (e.g. cleaning up after a customer demo),
# MAGIC run this in a Databricks notebook cell or from a local terminal:
# MAGIC
# MAGIC ```bash
# MAGIC cd "$REPO_ROOT" && databricks bundle destroy --target dev
# MAGIC ```
# MAGIC
# MAGIC This removes the Job + the synced bundle workspace files. The workshop's catalog,
# MAGIC schema, registered models, and serving endpoints are **separately** managed by
# MAGIC `scripts/reset_workshop.py` — run that if you want a fully clean slate.
