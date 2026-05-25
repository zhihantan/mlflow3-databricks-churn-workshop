# Module 10 — Capstone: Batch Score + Agent Drafted Outreach

Batch-score 100 customers via Module 4's deployed serving endpoint, rank the top-10 highest churn risk, invoke the deployed Module 8 agent on each to produce a personalized retention email, display the results in a styled table, and close on a productionization discussion (Jobs, scheduled workflows, human-in-the-loop with the Databricks Review App).

This module introduces no new APIs — it stitches together everything Modules 0–9 built. The end-to-end orchestration is the lesson.

**Concepts covered**
- End-to-end orchestration of classic ML + GenAI in a single flow
- Production patterns: scheduled Jobs, Slack alerting, human-in-the-loop review via the Databricks Review App
- Where Lakehouse Monitoring (Module 5) plugs in to close the feedback loop
- Discussion of governance (Unity AI Gateway) and cost (FMAPI rate limits, endpoint scale-to-zero)

**Prerequisites**
- Modules 4 and 8 have been run, and both their serving endpoints are in the `READY` state.

**Runtime target**: ~5 minutes.
**Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

**Notebook**: [`10_capstone.py`](./10_capstone.py)

---

> Status: scaffold stub.
