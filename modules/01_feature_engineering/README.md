# Module 01 — Feature Engineering in Unity Catalog

Build a UC feature table with point-in-time correctness for the churn label, using `FeatureEngineeringClient`. Output: a `customer_churn_features` feature table + a training-set view consumed by Module 2.

**Concepts covered**
- `FeatureEngineeringClient` from `databricks-feature-engineering`
- UC feature tables: `primary_keys` + `timeseries_columns` for time-aware joins
- `FeatureLookup` with `timestamp_lookup_key` for as-of-timestamp lookups
- Idempotent feature writes via `mode="merge"`

**Prerequisites**
- Module 0 (synthetic data) has been run.

**Runtime target**: ~4 minutes.
**Compute**: Serverless or DBR 17.3 LTS ML.

**Notebook**: [`01_feature_engineering.py`](./01_feature_engineering.py)

---

> Status: scaffold stub. The notebook is built next.
