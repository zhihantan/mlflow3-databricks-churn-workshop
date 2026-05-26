# Architecture

End-to-end view of what the workshop builds and how the pieces connect across the 10 modules.

---

## 1. End-to-end data flow

```mermaid
flowchart TB
    subgraph M0 [Module 0 — Synthetic data]
        C[customers]
        P[policies]
        CL[claims]
        PM[payments]
        T[support_tickets]
        S[customer_snapshots<br/>churned label]
    end

    subgraph M1 [Module 1 — Features]
        FT[customer_churn_features<br/>primary_keys customer_id snapshot_date<br/>timeseries_columns snapshot_date]
        TS[churn_training_set<br/>features + label]
    end

    C --> FT
    P --> FT
    CL --> FT
    PM --> FT
    T --> FT
    FT -->|FeatureLookup timestamp_lookup_key| TS
    S --> TS

    TS --> M2[Module 2<br/>LoggedModel: lgbm_baseline]
    TS --> M3[Module 3<br/>LoggedModel: lgbm_tuned]

    M2 -- model_id --> WS[workshop_state]
    M3 -- model_id --> WS
```

The single source of truth for inter-notebook handoffs is the `workshop_state` Delta table (small key/value table written via `MERGE INTO`).

---

## 2. Classic ML pipeline (Modules 2-5)

```mermaid
flowchart LR
    subgraph T [Modules 2-3]
        LR[LR baseline<br/>LoggedModel]
        LGB[LGBM baseline<br/>LoggedModel]
        TUN[Optuna 15 trials<br/>nested runs]
        FIN[LGBM tuned<br/>LoggedModel]
    end

    subgraph R [Module 4 — UC Registry]
        REG[bolttech_churn_model<br/>UC three-part name]
        AC[champion alias<br/>= LGBM tuned]
        ACH[challenger alias<br/>= LGBM baseline]
    end

    subgraph S [Module 4 — Model Serving]
        EP[Churn serving endpoint<br/>per-user]
        BATCH[Batch UDF<br/>via models:/...@champion]
    end

    subgraph MON [Module 5 — Monitoring]
        INF[Simulated inference table<br/>2-window drift]
        MO[scipy.stats drift<br/>+ mlflow.evaluate per window]
        DM[Drift metrics<br/>auto-generated Delta]
    end

    LR --> LGB --> TUN --> FIN
    FIN --> REG
    LR --> REG
    REG --> AC
    REG --> ACH
    AC --> EP
    REG --> BATCH
    INF --> MO --> DM
```

The **background-provisioning pattern** in Module 4 fires `serving_endpoints.create(...)` non-blocking, batch-scores via `mlflow.pyfunc.spark_udf` while the endpoint warms up, then `.result(timeout=...)`s at the end. Same pattern in Module 8 for the agent endpoint.

---

## 3. GenAI pipeline (Modules 6-9)

```mermaid
flowchart LR
    subgraph M6 [Module 6 — Tracing + Prompts]
        AL[mlflow.openai.autolog]
        FM[FMAPI chat call<br/>databricks-claude-haiku-4-5]
        PR[Prompt Registry<br/>churn_summary @production]
        VSE[VS endpoint kickoff<br/>background]
    end

    subgraph M7 [Module 7 — RAG]
        VSI[Vector Search Delta Sync index<br/>support_tickets<br/>managed embeddings gte-large-en]
        RAG[Traced RAG chain<br/>retrieve format generate]
        RP[churn_rag_qa @production]
    end

    subgraph M8 [Module 8 — Agent]
        AGT[ResponsesAgent<br/>OpenAI Agents SDK<br/>2 tools]
        TOOL1[churn_score_tool]
        TOOL2[tickets_tool]
        AGTUC[UC: bolttech_retention_agent]
        AGTEP[Deployed agent endpoint<br/>+ Review App]
    end

    subgraph M9 [Module 9 — Eval]
        ED[eval_dataset.py<br/>25 examples]
        EV[mlflow.genai.evaluate<br/>Correctness + Safety + Guidelines]
        IT[Prompt iteration loop<br/>v1 vs v2]
    end

    AL --> FM
    PR -.-> RAG
    VSE -.-> VSI
    VSI --> RAG --> RP
    RP -.-> AGT
    AGT --> TOOL1
    AGT --> TOOL2
    AGT --> AGTUC --> AGTEP
    AGT -.-> EV
    ED --> EV --> IT
```

Module 8's agent tools wire into Module 4's serving endpoint and Module 7's VS index respectively (via the `resources=[...]` declaration on `log_model`, which gives the deployed endpoint auto-auth to those Databricks resources).

---

## 4. End-to-end serving (Module 10 capstone)

```mermaid
sequenceDiagram
    participant N as Notebook
    participant SP as Spark
    participant ME as Module 4<br/>endpoint
    participant AE as Module 8<br/>agent endpoint
    participant VS as VS index
    participant FM as FMAPI

    N->>SP: load customer_churn_features
    N->>SP: spark_udf models:/...@champion
    SP->>ME: (in-process load, no REST call)
    SP-->>N: predicted_churn per customer
    N->>N: rank top-10
    loop For each top-10 customer
        N->>AE: predict request<br/>"Draft email for CUST_x"
        AE->>ME: churn_score_tool<br/>(REST POST features)
        ME-->>AE: churn probability
        AE->>VS: tickets_tool<br/>similarity_search filtered by cust
        VS-->>AE: top-k tickets
        AE->>FM: chat completion<br/>(via OpenAI Agents SDK)
        FM-->>AE: drafted email
        AE-->>N: ResponsesAgentResponse
    end
    N->>SP: persist capstone_retention_emails
```

---

## 5. Cross-cutting state

- **`workshop_state` Delta table** (per-user schema). Keys persisted: `lgbm_baseline_model_id`, `lgbm_tuned_model_id`, `lgbm_tuned_run_id`, `lgbm_tuned_test_auc`, `churn_endpoint_name`, `churn_model_uc_name`, `churn_champion_version`, `agent_model_id`, `agent_uc_name`, `agent_uc_version`, `agent_endpoint_name`, `agent_endpoint_url`, `agent_review_app_url`. Reading: `spark.table(STATE_TABLE).filter("key = '<k>'").select("value").first()[0]`.
- **MLflow experiment** (`/Users/<user>/mlflow3_workshop`). Every run, LoggedModel, trace, and eval result lands here.
- **Per-user UC schema** (`bolttech_workshop.churn_<sanitized_user>`). All Delta tables, the feature table, the inference table, and the workshop state table live here. Registered models also reference this schema in their three-part names.
- **Per-user serving endpoints** (`bolttech_churn_<user>`, plus the auto-named agent endpoint from `agents.deploy()`).
- **Per-user VS endpoint and index** (`bolttech_vs_<user>`, `<schema>.support_tickets_index`).
