"""Run in a configured Snowflake Notebook or Snowpark environment."""
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.registry import Registry


FEATURES = ["TENURE_MONTHS", "MONTHLY_FEE", "VISITS_LAST_30D", "DAYS_SINCE_LAST_VISIT", "SUPPORT_TICKETS_90D", "PAYMENT_FAILURES_90D", "DISCOUNT_PCT"]


def train_and_register(session):
    data = session.table("CHURN_FEATURE_STORE.TRAINING_DATA")
    train, test = data.random_split([.8, .2], seed=42)
    model = LogisticRegression(input_cols=FEATURES, label_cols=["CHURNED_60D"], output_cols=["PREDICTION"])
    model.fit(train)
    registry = Registry(session=session, database_name=session.get_current_database(), schema_name="MODEL_REGISTRY")
    version = registry.log_model(
        model=model,
        model_name="MEMBER_CHURN_MODEL",
        version_name="V1",
        sample_input_data=test.select(FEATURES).limit(100),
        metrics={"business_purpose": "60-day subscription churn prioritisation"},
        comment="Owner: Customer Analytics; approved threshold required before production",
    )
    return version
