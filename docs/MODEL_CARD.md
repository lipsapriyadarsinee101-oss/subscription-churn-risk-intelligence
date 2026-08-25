# Model Card — Member Churn Model v1

## Intended use
Prioritise members for retention outreach by estimating cancellation within 60 days. Scores support human decisions and must not be used for eligibility, pricing discrimination or other irreversible actions.

## Target and prediction unit
Binary target: `churned_60d`. Prediction unit: one active member at a defined snapshot date.

## Features
Tenure, monthly fee, 30-day visits, recency, 90-day support tickets, 90-day payment failures, discount, plan, contract and region. Post-cancellation information is excluded.

## Validation
The newest 20% of snapshots form the test set. Report ROC-AUC, PR-AUC, Brier score, precision, recall and F1. Review metrics by plan and region before release.

## Limitations
The included dataset is synthetic and proves the engineering workflow, not real-world performance. Thresholds, protected-group reviews and retention economics require validation on authorised production data.

## Monitoring
Monitor feature PSI, prediction distribution, cohort volume, calibration, realised churn, latency, missingness and intervention uplift. Retrain after material drift or an approved business change.
