from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build import FEATURES
from src.scoring.cohorts import MODEL_VERSION, score_members


class MemberRequest(BaseModel):
    member_id: str
    tenure_months: int = Field(ge=0)
    monthly_fee: float = Field(gt=0)
    visits_last_30d: int = Field(ge=0)
    days_since_last_visit: float = Field(ge=0)
    support_tickets_90d: int = Field(ge=0)
    payment_failures_90d: int = Field(ge=0)
    discount_pct: float = Field(ge=0, le=100)
    plan_type: str
    contract_type: str
    region: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path("artifacts/churn_model.joblib")
    app.state.model = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(title="Churn Risk Scoring API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy" if app.state.model is not None else "model_not_trained", "model_version": MODEL_VERSION}


@app.post("/v1/score")
def score(member: MemberRequest):
    if app.state.model is None:
        raise HTTPException(503, "Run python -m src.training.train before scoring")
    frame = pd.DataFrame([{**member.model_dump(), "snapshot_date": pd.Timestamp.utcnow()}])
    probability = app.state.model.predict_proba(frame[FEATURES])[:, 1]
    return score_members(frame, probability).iloc[0].to_dict()
