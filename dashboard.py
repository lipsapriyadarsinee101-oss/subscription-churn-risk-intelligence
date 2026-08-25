from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Churn Risk Intelligence", page_icon="📉", layout="wide")
st.title("Subscription Churn Risk Intelligence")
st.caption("Member-level risk, customer value and retention activation")

scores_path = Path("artifacts/cdp_member_scores.csv")
metrics_path = Path("artifacts/metrics.json")
if not scores_path.exists():
    st.warning("Generate data and train the model first: `python -m src.data.generate && python -m src.training.train`")
    st.stop()

scores = pd.read_csv(scores_path)
metrics = json.loads(metrics_path.read_text())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Scored members", f"{len(scores):,}")
c2.metric("High risk", f"{(scores.risk_cohort == 'HIGH').mean():.1%}")
c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
c4.metric("Value at risk", f"€{scores.loc[scores.risk_cohort == 'HIGH', 'predicted_clv'].sum():,.0f}")

left, right = st.columns(2)
left.subheader("Risk cohort distribution")
left.bar_chart(scores.risk_cohort.value_counts())
right.subheader("Probability distribution")
right.bar_chart(scores.churn_probability.value_counts(bins=10).sort_index())
st.subheader("Priority retention queue")
st.dataframe(scores.sort_values(["risk_cohort", "predicted_clv"], ascending=[True, False]), use_container_width=True, hide_index=True)
