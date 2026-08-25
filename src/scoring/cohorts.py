from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features.build import calculate_clv


MODEL_VERSION = "churn_hgb_v1"


def risk_cohort(probability: float) -> str:
    if probability >= 0.30:
        return "HIGH"
    if probability >= 0.15:
        return "MEDIUM"
    return "LOW"


def score_members(frame: pd.DataFrame, probabilities) -> pd.DataFrame:
    output = frame[["member_id", "snapshot_date"]].copy()
    output["churn_probability"] = np.asarray(probabilities).round(4)
    output["risk_cohort"] = output["churn_probability"].map(risk_cohort)
    output["predicted_clv"] = calculate_clv(frame, probabilities)
    actions = {"HIGH": "Priority outreach + tailored offer", "MEDIUM": "Engagement campaign", "LOW": "Loyalty reinforcement"}
    output["recommended_action"] = output["risk_cohort"].map(actions)
    output["model_version"] = MODEL_VERSION
    output["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    return output
