import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_members(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    monthly_fee = rng.normal(42, 14, rows).clip(10, 120).round(2)
    visits = rng.poisson(7, rows)
    days_since = rng.gamma(2.2, 7, rows).clip(0, 90).round(1)
    tickets = rng.poisson(0.7, rows)
    payment_failures = rng.binomial(3, 0.08, rows)
    discount = rng.choice([0, 5, 10, 15, 20], rows, p=[.45, .15, .2, .12, .08])
    plan = rng.choice(["basic", "plus", "premium"], rows, p=[.48, .34, .18])
    contract = rng.choice(["monthly", "annual"], rows, p=[.68, .32])
    region = rng.choice(["north", "south", "east", "west"], rows)
    score = (-2.5 + .055 * days_since + .48 * payment_failures + .16 * tickets - .035 * visits - .018 * tenure + .7 * (contract == "monthly") - .015 * discount)
    probability = 1 / (1 + np.exp(-score))
    churned = rng.binomial(1, probability)
    snapshot = pd.Timestamp("2026-07-31") - pd.to_timedelta(rng.integers(0, 365, rows), unit="D")
    return pd.DataFrame({
        "member_id": [f"M{i:06d}" for i in range(1, rows + 1)],
        "snapshot_date": snapshot,
        "tenure_months": tenure,
        "monthly_fee": monthly_fee,
        "visits_last_30d": visits,
        "days_since_last_visit": days_since,
        "support_tickets_90d": tickets,
        "payment_failures_90d": payment_failures,
        "discount_pct": discount,
        "plan_type": plan,
        "contract_type": contract,
        "region": region,
        "churned_60d": churned,
    }).sort_values("snapshot_date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path("data/generated/members.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_members(args.rows, args.seed)
    frame.to_csv(output, index=False)
    print(f"Created {len(frame):,} member snapshots at {output}; churn rate={frame.churned_60d.mean():.1%}")


if __name__ == "__main__":
    main()
