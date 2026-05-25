# Module 04 — UC Model Registry + Model Serving

Register the Module 3 tuned LightGBM model in Unity Catalog under a three-part name, set `@champion` / `@challenger` aliases, kick off a serving endpoint in cell 1 (background provisioning), demonstrate batch scoring via `models:/...` while the endpoint provisions, then query the live REST endpoint at the end.

**Concepts covered**
- `mlflow.set_registry_uri("databricks-uc")` + `mlflow.register_model(model_uri="models:/<model_id>", name="<cat>.<sch>.bolttech_churn_model")`
- Model aliases via `MlflowClient().set_registered_model_alias(...)` — `@champion` and `@challenger`
- Batch scoring with `mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}@champion")`
- `WorkspaceClient.serving_endpoints.create_and_wait(...)` with `EndpointCoreConfigInput` + `ServedEntityInput`
- Endpoint query via `mlflow.deployments.get_deploy_client("databricks").predict(...)`
- Endpoint cold-start mitigation: provision in cell 1, do batch-scoring work while it warms up, query at the end (the *background-provisioning pattern* used again in Module 8)

**Prerequisites**
- Modules 0, 1, 2, 3 have been run.

**Runtime target**: ~8 minutes (endpoint cold-start ~5-7 min absorbed by parallel work).
**Compute**: Serverless ML (Beta) or DBR 17.3 LTS ML.

**Notebook**: [`04_registry_and_serving.py`](./04_registry_and_serving.py)

---

> Status: scaffold stub.
