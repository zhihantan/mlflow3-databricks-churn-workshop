# Instructor Guide

Practical notes for facilitators running this workshop live. Read in conjunction with `PLAN.md` (curriculum + technical decisions) and `VERIFICATION.md` (per-module checks + flagged items).

---

## Pre-workshop checklist

**One week before**
- [ ] Confirm the workspace has UC enabled, FMAPI enabled, and Vector Search enabled.
- [ ] Confirm each participant has been granted (a) read on the metastore, (b) `CREATE CATALOG` OR an admin has pre-created `bolttech_workshop`, (c) `CREATE SCHEMA` on `bolttech_workshop`, (d) ability to create serving endpoints and VS endpoints.
- [ ] Decide on compute: Serverless preferred for fast start; classic DBR 17.3 LTS ML cluster as fallback. If using classic, provision a cluster per participant or a shared cluster with sufficient capacity (recommend Standard_DS4_v2 / m5.xlarge minimum).
- [ ] Skim `VERIFICATION.md` §3 — the flagged items — and decide whether you want to live-verify any of them before the session.
- [ ] If group > 10 participants on a single workspace, plan for FMAPI rate-limit contention. Module 9 makes ~75 judge calls × N participants concurrently — at 10 participants you're at ~750 calls in a few minutes, well within the 20K OTPM limit, but stagger Module 9 start times if you go higher.

**Day of**
- [ ] Distribute the git clone URL to participants.
- [ ] (Optional) Pre-warm: instructor runs Module 6's VS endpoint kickoff and Module 4's churn endpoint provisioning ~10 min before the session starts. This saves ~10 min of cumulative cold-start time across the room.
- [ ] (Optional) Have a Lakeview dashboard or SQL Alert pre-configured against the `churn_drift_metrics` table ready to show participants what production monitoring looks like layered on top of M5's output.

---

## Per-module talking points

### Module 0 — Setup & synthetic data (~3 min)
**Anchor:** "Everything is reproducible because everything is seeded."
- 30s framing: "This generates 20k synthetic bolttech customers — fully fake, fully deterministic. Re-running gives the exact same data."
- Worth pointing to: the `SYNTHETIC_SEED=42` constant and the layered `_expand_payments` / churn-probability logic.
- Common pitfall: participants forget to run setup before opening Module 1. If they hit "table does not exist" in M1, send them back to Module 0.

### Module 1 — Feature engineering (~4 min)
**Anchor:** "Point-in-time correctness is what separates a feature store from a SQL view."
- 30s framing: "Feature Engineering in UC isn't just storage — it's the type system that prevents target leakage."
- Highlight: `timeseries_columns='snapshot_date'` on the feature table + `timestamp_lookup_key='snapshot_date'` on the `FeatureLookup`. Without that pairing, the join would happily pull future-dated features and silently leak the label.
- Common pitfall: customers without any events get NULL aggregates. The `.fillna(0, subset=[...])` handles it but worth flagging.

### Module 2 — Experiment tracking & LoggedModel (~5 min)
**Anchor:** "`LoggedModel` is THE big MLflow 3 change. Everything else builds on it."
- 30s framing: "In MLflow 2 a model was an artifact inside a run. In MLflow 3 it's a first-class entity with its own ID, URI, and lifecycle."
- The visceral moment: open the experiment UI after the cells run. The Artifacts tab no longer shows model files; the Models tab does, with two separate model pages. Point out `models:/<model_id>` URI as the new canonical address.
- Have the §10 markdown table on the screen when discussing breaking changes. The three big ones are `name=` vs `artifact_path=`, run-scoped artifacts being gone, and `baseline_model=` removal from `mlflow.evaluate`.

### Module 3 — Tuning & evaluation (~5 min)
**Anchor:** "`model_id=` on `mlflow.evaluate` is what gives you lineage from eval to production."
- 30s framing: "Optuna for the tuning loop, `mlflow.evaluate` for the audit trail, and the new `model_id=` kwarg ties them to a specific LoggedModel."
- The business-metric pattern (`expected_retention_value`) is the part participants remember. Walk through the formula.
- Note `baseline_model=` is gone in MLflow 3 — show the side-by-side comparison we do manually.

### Module 4 — Registry + serving (~7-8 min) — the runtime-critical module
**Anchor:** "The background-provisioning pattern is what makes a 7-min module fit a 60-min workshop."
- 30s framing: "Serving endpoint cold-starts take 5-7 min. We absorb that wait by doing all the registration + batch-scoring work while the endpoint provisions."
- Cell-by-cell narration: cell 5 fires the endpoint (non-blocking), cells 6-8 do useful work, cell 9 polls for readiness.
- This is also where Q3 from PLAN.md was locked — the workshop **really** deploys, and Module 8 reuses the same pattern.

### Module 5 — Monitoring (~2-3 min)
**Anchor:** "Drift is a Delta table you wrote with one `saveAsTable` — same SQL, same governance, same alerts as everything else."
- 30s framing: "Two complementary checks — input drift via scipy, prediction drift via `mlflow.evaluate`. Both land in MLflow + Delta. No separate observability stack."
- The simulated drift on `payment_failures_60d` (×2 in window 2) is intentionally exaggerated so the window-over-window deltas printed at the bottom of cell 7 are unmistakable.
- Pitch the production wiring: "Schedule this notebook nightly. SQL Alert when `features_with_drift > 0`. Alert triggers the retraining Job. That's the full MLOps feedback loop in three Databricks-native pieces."

### Module 6 — Tracing + Prompt Registry (~3 min)
**Anchor:** "Three primitives — `autolog`, `register_prompt`, `@production` — that every later GenAI module reuses."
- Important: this module kicks off the VS endpoint provisioning in cell 2. **Don't skip cell 2**, or Module 7 will be slow.
- Watch participants try to use single curly braces in templates — `{var}` won't work, it must be `{{var}}`. Flag it.
- Open the MLflow Traces tab after the chat call. The trace tree with token counts and latency is the visual that sells tracing.

### Module 7 — RAG (~6-8 min)
**Anchor:** "Managed embeddings + Delta Sync means zero embed-pipeline plumbing."
- 30s framing: "You point at a Delta column, you point at an embedding model, Vector Search handles the rest."
- If the VS endpoint isn't ONLINE yet (because Module 6 cell 2 was skipped or interrupted), this is where the workshop visibly stalls. Pre-warming saves participants here.
- The traced RAG chain (`retrieve_tickets` → `format_context` → `churn_rag_qa`) is what they'll show their team in the post-mortem screenshot.

### Module 8 — Retention agent (~8-9 min) — the most ambitious module
**Anchor:** "ResponsesAgent + Models-from-Code + agents.deploy() — the full production path in one notebook."
- 30s framing: "Two tools, one agent class, one `agents.deploy()` call. Everything else is plumbing MLflow handles for us."
- Walk through `agent.py` in the IDE alongside the driver notebook. Participants need to see both files.
- The `resources=[...]` declaration is the single magical line — it's what makes the deployed agent able to reach the M4 endpoint and the M7 VS index without manual token plumbing.
- Background-provisioning pattern from Module 4 reappears here. Reinforce the pattern.

### Module 9 — GenAI evaluation (~5-6 min)
**Anchor:** "Evaluation is a Run with judges instead of metrics."
- 30s framing: "`mlflow.genai.evaluate` is the unified entry point — built-in judges, custom judges, custom predict_fns, all in one call."
- The prompt iteration demo (v1 vs v2) is the take-home. Show the comparison DataFrame and walk through which scorer moved.
- Caveat: this module hits FMAPI hard. If you see judge-call timeouts, you've hit OTPM. Stagger module starts in large groups.

### Module 10 — Capstone (~3-5 min)
**Anchor:** "Everything you built, in one orchestration."
- This module is fast — the work was done in the earlier modules. Use the saved time for Q&A.
- If the M8 deployed endpoint never became READY, Module 10 falls back to the local agent automatically. Don't panic; flag the fallback message.
- The productionization markdown table is the natural seg into a closing Q&A.

---

## Common participant pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Forgot to run Module 0 | `Table does not exist` in M1 | Send them back to M0 |
| Skipped Module 6 cell 2 | M7 stalls waiting for VS endpoint | Re-run Module 6 cell 2, wait 5 min, then resume M7 |
| Single curly braces in prompt template | `KeyError: 'var'` from `prompt.format(...)` | Reinforce `{{var}}` syntax |
| Tried to run M9 before M8 finished | `Model not found` in M9 | M8 must complete logging (cell 3); local-test cells (5-6) need to have run |
| Did NOT %pip install in each module | Stale `mlflow` from DBR 17.3 LTS ML (3.0.1) lacks 3.12 features | Always run the `%pip install` cell first; verify `mlflow.__version__` if suspicious |
| Missing `scipy` in M5 | `ModuleNotFoundError: scipy` | scipy is preinstalled on DBR ML LTS + Serverless ML — if missing for any reason, add it to the pip install cell at the top of the module |
| Hit FMAPI OTPM in M9 | Long judge-call timeouts | Wait 60s, re-run the failed eval. For groups, stagger start times. |

---

## Pacing notes

- The workshop is designed for **45-50 minutes of code execution + 10-15 minutes of narration and Q&A** = 60 min total. If you have a 75-min slot, lean into Q&A in Modules 4, 7, and 10.
- **Cold-start absorption** matters. The two pre-warm wait points (M4 endpoint, M7 VS endpoint) account for ~12 min cumulative wait, absorbed by parallel work. Don't skip the cells that "look idle" during a cold-start — that's by design.
- **Best place to pause for Q&A**: end of Module 2 (LoggedModel is the deepest single concept), and end of Module 8 (after the full GenAI stack is built).

---

## What's NOT in the workshop (questions to anticipate)

- **No Unity AI Gateway hands-on.** Mentioned only as a closing-slide bullet in Module 10. If asked, point at https://docs.databricks.com/aws/en/ai-gateway/ and offer a separate gateway-focused session.
- **No A/B testing on serving traffic split.** The `@champion`/`@challenger` aliases are set, but we don't route % of traffic in Module 4. Mention `traffic_config` on the endpoint as the production path.
- **No CI/CD.** This is a notebook workshop. For Jobs / repos-driven deployment, point at Databricks Asset Bundles (DABs).
- **No multi-snapshot point-in-time training.** Module 1 has one snapshot date. In production you'd have many — point at the time-series feature lookup docs.
- **No real Review App walk-through.** Module 8 surfaces the Review App URL, but driving the UI isn't in the workshop scope. Optional add-on.

---

## Troubleshooting cheat-sheet (handout)

For each participant, this cheat-sheet (also in `README.md` §Troubleshooting) covers the top 5 expected failure modes. Print or share as a tab in your slide deck.

1. `ModuleNotFoundError: No module named 'config.workshop_config'` → clone via Repos UI, not file upload.
2. `RESOURCE_DOES_NOT_EXIST` on endpoint query → wait 2-5 more min, re-run polling cell.
3. `PERMISSION_DENIED` on `CREATE CATALOG` → ask admin to run the catalog create once.
4. VS index stuck in `PROVISIONING` → check Catalog Explorer → Compute → Vector Search; delete + retry.
5. `agents.deploy()` kwarg error → already wrapped in `try/except`; if both branches fail, run `%pip show databricks-agents`.
