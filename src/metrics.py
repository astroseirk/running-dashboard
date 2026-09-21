"""Derived metrics computed from the cleaned activity table."""
import pandas as pd

from src.data import HR_ZONE_COLS

# Standard race distances (km) used to pick out "best effort" runs.
PB_DISTANCES = {
    "5K": (4.8, 5.3),
    "10K": (9.7, 10.3),
    "Half Marathon": (20.5, 21.6),
    "Marathon": (41.5, 43.5),
}


def format_pace(pace_min_per_km: float) -> str:
    if pd.isna(pace_min_per_km):
        return "-"
    minutes = int(pace_min_per_km)
    seconds = round((pace_min_per_km - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d} /km"


def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby("week")
        .agg(
            distance_km=("distance_km", "sum"),
            runs=("id", "count"),
            training_load=("icu_training_load", "sum"),
            avg_pace=("pace_min_per_km", "mean"),
        )
        .reset_index()
    )
    return weekly


def fitness_fatigue_trend(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["start_date_local", "icu_fitness", "icu_fatigue"]
    trend = df[cols].dropna(subset=["icu_fitness", "icu_fatigue"]).copy()
    trend["form"] = trend["icu_fitness"] - trend["icu_fatigue"]
    return trend


def hr_zone_totals(df: pd.DataFrame, by: str = "month") -> pd.DataFrame:
    present = [c for c in HR_ZONE_COLS if c in df.columns]
    grouped = df.groupby(by)[present].sum().reset_index()
    hours = grouped[present] / 3600
    hours.columns = [f"Zone {c[4]}" for c in present]
    return pd.concat([grouped[[by]], hours], axis=1)


def personal_bests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (low, high) in PB_DISTANCES.items():
        candidates = df[(df["distance_km"] >= low) & (df["distance_km"] <= high)]
        if candidates.empty:
            continue
        best = candidates.loc[candidates["pace_min_per_km"].idxmin()]
        rows.append(
            {
                "Distance": label,
                "Date": best["start_date_local"].date(),
                "Run": best["name"],
                "Pace": format_pace(best["pace_min_per_km"]),
                "Time (min)": round(best["moving_time_min"], 1),
            }
        )
    return pd.DataFrame(rows)


EASY_EFFORT_RPE_MAX = 3


def easy_effort_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Runs at RPE <= 3, regardless of title. More representative of true easy
    effort than the name-based 'Easy Run' tag, which only exists on a
    minority of runs that happened to be titled that way.
    """
    if "icu_rpe" not in df.columns:
        return df.iloc[0:0]
    cohort = df[df["icu_rpe"] <= EASY_EFFORT_RPE_MAX].dropna(subset=["average_heartrate", "pace_min_per_km"]).copy()
    cohort["speed_kmh"] = 60 / cohort["pace_min_per_km"]
    cohort["efficiency"] = cohort["speed_kmh"] / cohort["average_heartrate"]
    return cohort


def easy_effort_monthly(df: pd.DataFrame) -> pd.DataFrame:
    cohort = easy_effort_runs(df)
    if cohort.empty:
        return cohort
    monthly = (
        cohort.groupby(cohort["start_date_local"].dt.to_period("M"))
        .agg(
            avg_hr=("average_heartrate", "mean"),
            avg_pace=("pace_min_per_km", "mean"),
            efficiency=("efficiency", "mean"),
            runs=("id", "count"),
        )
        .reset_index()
    )
    monthly["start_date_local"] = monthly["start_date_local"].dt.to_timestamp()
    return monthly


def easy_effort_half_split(df: pd.DataFrame) -> dict | None:
    cohort = easy_effort_runs(df)
    if len(cohort) < 4:
        return None
    cohort = cohort.sort_values("start_date_local")
    mid = cohort["start_date_local"].median()
    first = cohort[cohort["start_date_local"] <= mid]
    second = cohort[cohort["start_date_local"] > mid]
    return {
        "first": {
            "n": len(first),
            "avg_hr": first["average_heartrate"].mean(),
            "avg_pace": first["pace_min_per_km"].mean(),
            "efficiency": first["efficiency"].mean(),
        },
        "second": {
            "n": len(second),
            "avg_hr": second["average_heartrate"].mean(),
            "avg_pace": second["pace_min_per_km"].mean(),
            "efficiency": second["efficiency"].mean(),
        },
    }


def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total_runs": len(df),
        "total_distance_km": df["distance_km"].sum(),
        "total_time_hours": df["moving_time"].sum() / 3600,
        "avg_pace": format_pace(df["pace_min_per_km"].mean()),
        "date_range": (df["start_date_local"].min().date(), df["start_date_local"].max().date()),
    }
