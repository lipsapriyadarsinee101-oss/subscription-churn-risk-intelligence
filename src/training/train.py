import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, validate
from src.scoring.cohorts import score_members


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("features", preprocessing), ("model", LogisticRegression(max_iter=1000, C=.8, random_state=42))])


def train(data_path: str = "data/generated/members.csv") -> dict:
    frame = pd.read_csv(data_path, parse_dates=["snapshot_date"]).sort_values("snapshot_date")
    validate(frame)
    split = int(len(frame) * .8)
    train_frame, test_frame = frame.iloc[:split], frame.iloc[split:]
    model = build_pipeline().fit(train_frame[FEATURES], train_frame["churned_60d"])
    probability = model.predict_proba(test_frame[FEATURES])[:, 1]
    decision_threshold = .20
    prediction = (probability >= decision_threshold).astype(int)
    report = classification_report(test_frame["churned_60d"], prediction, output_dict=True, zero_division=0)
    metrics = {
        "roc_auc": round(roc_auc_score(test_frame["churned_60d"], probability), 4),
        "pr_auc": round(average_precision_score(test_frame["churned_60d"], probability), 4),
        "brier_score": round(brier_score_loss(test_frame["churned_60d"], probability), 4),
        "precision": round(report["1"]["precision"], 4), "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4), "confusion_matrix": confusion_matrix(test_frame["churned_60d"], prediction).tolist(),
        "train_rows": len(train_frame), "test_rows": len(test_frame), "decision_threshold": decision_threshold,
    }
    artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True)
    joblib.dump(model, artifacts / "churn_model.joblib")
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    driver_names = model.named_steps["features"].get_feature_names_out()
    pd.DataFrame({"feature": driver_names, "coefficient": model.named_steps["model"].coef_[0]}).assign(
        absolute_impact=lambda data: data.coefficient.abs()
    ).sort_values("absolute_impact", ascending=False).to_csv(artifacts / "global_churn_drivers.csv", index=False)
    score_members(test_frame, probability).to_csv(artifacts / "cdp_member_scores.csv", index=False)
    train_frame[NUMERIC_FEATURES].describe().to_json(artifacts / "training_profile.json")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
