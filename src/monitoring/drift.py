import numpy as np


def population_stability_index(expected, actual, buckets: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 2:
        return 0.0
    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return round(float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))), 4)


def drift_status(psi: float) -> str:
    return "ALERT" if psi >= .25 else "WATCH" if psi >= .10 else "STABLE"


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    psi = population_stability_index(rng.normal(0, 1, 1000), rng.normal(.35, 1.1, 1000))
    print({"psi": psi, "status": drift_status(psi)})
