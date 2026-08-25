import pandas as pd


NUMERIC_FEATURES = [
    "tenure_months", "monthly_fee", "visits_last_30d", "days_since_last_visit",
    "support_tickets_90d", "payment_failures_90d", "discount_pct",
]
CATEGORICAL_FEATURES = ["plan_type", "contract_type", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate(frame: pd.DataFrame) -> None:
    missing = set(FEATURES + ["member_id", "snapshot_date"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["member_id"].duplicated().any():
        raise ValueError("member_id must be unique per scoring snapshot")
    if (frame[NUMERIC_FEATURES] < 0).any().any():
        raise ValueError("Numeric behavioural values cannot be negative")


def calculate_clv(frame: pd.DataFrame, churn_probability, gross_margin: float = 0.72) -> pd.Series:
    expected_months = (1 / pd.Series(churn_probability, index=frame.index).clip(lower=0.03)).clip(upper=36)
    value = frame["monthly_fee"] * gross_margin * expected_months
    return value.round(2)
