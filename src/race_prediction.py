"""Out-of-sample race-time prediction via Riegel's power law.

For a target race, fit `time = a * distance^b` (log-log linear regression)
using only hard-effort training runs strictly before that race's date, then
extrapolate to the race distance. The race itself is never in the fit.
"""
import numpy as np
import pandas as pd

RACES = {
    "Marathon 2024": {"date": "2024-05-05", "distance_km": 42.195},
    "Half Marathon 2024": {"date": "2024-09-15", "distance_km": 21.0975},
    "Marathon 2025": {"date": "2025-05-11", "distance_km": 42.195},
}

HARD_EFFORT_RPE_MIN = 6
HARD_EFFORT_TYPES = ("Tempo", "Intervals", "Race")
MIN_POINTS_TO_FIT = 4


def hard_effort_points(df: pd.DataFrame, before_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training runs strictly before `before_date`, and the subset of those
    that represent a genuine hard/near-max effort (used for the fit).
    """
    cutoff = pd.Timestamp(before_date)
    train = df[df["start_date_local"] < cutoff].copy()
    hard = train[(train["workout_type"].isin(HARD_EFFORT_TYPES)) | (train["icu_rpe"] >= HARD_EFFORT_RPE_MIN)]
    hard = hard.dropna(subset=["distance", "moving_time"])
    hard = hard[(hard["distance"] > 0) & (hard["moving_time"] > 0)]
    return hard, train


def fit_riegel(hard: pd.DataFrame) -> tuple[float, float]:
    """Fit time = a * distance^b via log-log OLS. Returns (a, b)."""
    d = hard["distance"].values.astype(float)
    t = hard["moving_time"].values.astype(float)
    b, log_a = np.polyfit(np.log(d), np.log(t), 1)
    return np.exp(log_a), b


def predict_seconds(a: float, b: float, distance_km: float) -> float:
    return a * (distance_km * 1000) ** b


def race_report(df: pd.DataFrame, race_label: str) -> dict:
    meta = RACES[race_label]
    race_date, dist_km = meta["date"], meta["distance_km"]

    actual_rows = df[df["start_date_local"].dt.date == pd.Timestamp(race_date).date()]
    if actual_rows.empty:
        return {"race_label": race_label, "found": False}
    actual = actual_rows.iloc[0]

    hard, train = hard_effort_points(df, race_date)
    result = {
        "race_label": race_label,
        "found": True,
        "race_date": pd.Timestamp(race_date).date(),
        "distance_km": dist_km,
        "actual_seconds": float(actual["moving_time"]),
        "actual_name": actual["name"],
        "n_training_runs": len(train),
        "hard_effort_points": hard,
        "fit_ok": len(hard) >= MIN_POINTS_TO_FIT,
    }
    if not result["fit_ok"]:
        return result

    a, b = fit_riegel(hard)
    pred_sec = predict_seconds(a, b, dist_km)
    result.update(
        {
            "a": a,
            "b": b,
            "predicted_seconds": pred_sec,
            "min_duration_min": hard["moving_time"].min() / 60,
            "max_duration_min": hard["moving_time"].max() / 60,
            "error_seconds": pred_sec - result["actual_seconds"],
            "error_pct": 100 * (pred_sec - result["actual_seconds"]) / result["actual_seconds"],
        }
    )
    return result
