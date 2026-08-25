CREATE SCHEMA IF NOT EXISTS CHURN_FEATURE_STORE;

CREATE OR REPLACE DYNAMIC TABLE CHURN_FEATURE_STORE.MEMBER_FEATURES
TARGET_LAG = '1 day'
WAREHOUSE = ANALYTICS_WH
AS
SELECT
    m.member_id,
    CURRENT_DATE() AS snapshot_date,
    DATEDIFF('month', m.join_date, CURRENT_DATE()) AS tenure_months,
    m.monthly_fee,
    COUNT_IF(a.event_date >= DATEADD('day', -30, CURRENT_DATE())) AS visits_last_30d,
    DATEDIFF('day', MAX(a.event_date), CURRENT_DATE()) AS days_since_last_visit,
    COUNT(DISTINCT CASE WHEN s.created_at >= DATEADD('day', -90, CURRENT_DATE()) THEN s.ticket_id END) AS support_tickets_90d,
    COUNT_IF(p.status = 'FAILED' AND p.payment_date >= DATEADD('day', -90, CURRENT_DATE())) AS payment_failures_90d,
    m.discount_pct,
    m.plan_type,
    m.contract_type,
    m.region
FROM RAW.MEMBERS m
LEFT JOIN RAW.ACTIVITY a USING (member_id)
LEFT JOIN RAW.SUPPORT_TICKETS s USING (member_id)
LEFT JOIN RAW.PAYMENTS p USING (member_id)
GROUP BY ALL;

CREATE TABLE IF NOT EXISTS ANALYTICS.CDP_MEMBER_CHURN_SCORES (
    member_id VARCHAR, snapshot_date DATE, churn_probability FLOAT,
    risk_cohort VARCHAR, predicted_clv NUMBER(12,2), recommended_action VARCHAR,
    model_version VARCHAR, scored_at_utc TIMESTAMP_TZ
);

-- Optional Snowflake ML Functions baseline for model comparison:
-- CREATE SNOWFLAKE.ML.CLASSIFICATION CHURN_BASELINE(
--   INPUT_DATA => SYSTEM$REFERENCE('TABLE', 'TRAINING_DATA'), TARGET_COLNAME => 'CHURNED_60D'
-- );
