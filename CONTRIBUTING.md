# Contributing

This is a Databricks Field Engineering workshop repo. Pull requests welcome under the following guidelines:

## Code
- Verify every MLflow / Databricks API call against the **current** docs (training data is stale on MLflow 3 and the Agent Framework — confirm signatures before submitting).
- Add an inline `# Ref: <docs URL>` comment above any non-obvious MLflow / Databricks call.
- Keep notebooks in **Databricks notebook source format** (`.py` with `# Databricks notebook source` header + `# COMMAND ----------` cells) so they render correctly via Git folders.
- Read identifiers from `config/workshop_config.py` — never hardcode catalog, schema, model, or endpoint names in module notebooks.
- Make every notebook idempotent (`CREATE OR REPLACE`, `try`/`except` around endpoint/index creation, etc.).

## Runtime budget
- The full workshop must run in under **60 minutes** of participant code execution on a fresh workspace (target ≤45 min — see `PLAN.md` §4).
- Per-module budgets are tracked in `VERIFICATION.md`; if you blow a budget, flag it in the PR description.

## Testing before opening a PR
- Clone the repo into a Databricks workspace via Git folders.
- Run `setup/00_setup_and_synthetic_data.py`, then all `modules/NN_*/NN_*.py` notebooks top-to-bottom.
- Optionally: run `scripts/reset_workshop.py` afterwards to confirm clean tear-down.

## Issues
File an issue with: the notebook path, the DBR version (or Serverless ML environment version), and the full traceback.
