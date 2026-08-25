# Production Runbook

## Daily scoring
1. Validate source freshness and row counts.
2. Refresh point-in-time feature views.
3. Score active members with the approved model version.
4. Apply cohort thresholds and CLV calculation.
5. Run uniqueness, range and null checks.
6. Merge results into the CDP activation table.
7. Publish Power BI refresh status and audit metadata.

## Incident response
If quality gates fail, stop activation, retain the previous valid score table, record the failed run ID, and notify Customer Analytics and the platform owner. Rollback means restoring the previous Model Registry version and rerunning scoring.

## Ownership
Customer Analytics owns target definition and action thresholds. Data Engineering owns source SLAs and feature pipelines. ML Engineering owns training, registry, serving and monitoring. CRM Operations owns campaign execution and outcome feedback.

## Retraining
Retraining is triggered quarterly, after PSI >= 0.25 on a critical feature, after a significant calibration decline, or after a change to products, pricing or cancellation policy. Promotion requires offline validation and stakeholder approval.
