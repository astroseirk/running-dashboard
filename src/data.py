"""Load and clean the intervals.icu activity export."""
import io
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "activities.csv"

HR_ZONE_COLS = [f"hr_z{i}_secs" for i in range(1, 8)]

WORKOUT_KEYWORDS = [
    ("Race", ["race"]),
    ("Intervals", ["interval", "repeat", "400", "800", "vo2"]),
    ("Tempo", ["tempo"]),
    ("Long Run", ["long run"]),
    ("Easy Run", ["easy"]),
]


def _classify_workout(name: str) -> str:
    lowered = (name or "").lower()
    for label, keywords in WORKOUT_KEYWORDS:
        if any(k in lowered for k in keywords):
            return label
    return "Other"


def _unwrap_double_quoted_rows(text: str) -> str:
    """intervals.icu exports each data row wrapped in one extra layer of CSV
    quoting (the whole line is a quoted field, internal quotes doubled).
    Strip that outer layer so the remaining text is plain, once-quoted CSV.
    """
    lines = text.splitlines()
    if not lines:
        return text
    header, *rows = lines
    fixed_rows = []
    for row in rows:
        if row.startswith('"') and row.endswith('"'):
            row = row[1:-1].replace('""', '"')
        fixed_rows.append(row)
    return "\n".join([header, *fixed_rows])


def load_activities(path: Path = DATA_PATH) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig")
    df = pd.read_csv(io.StringIO(_unwrap_double_quoted_rows(raw)))
    df = df[df["type"] == "Run"].copy()

    df["start_date_local"] = pd.to_datetime(df["start_date_local"], errors="coerce")
    df = df.dropna(subset=["start_date_local", "distance", "moving_time"])

    df["distance_km"] = df["distance"] / 1000
    df["moving_time_min"] = df["moving_time"] / 60
    df["pace_min_per_km"] = df["moving_time_min"] / df["distance_km"].replace(0, pd.NA)

    df["week"] = df["start_date_local"].dt.to_period("W").apply(lambda p: p.start_time)
    df["month"] = df["start_date_local"].dt.to_period("M").apply(lambda p: p.start_time)

    df["workout_type"] = df["name"].apply(_classify_workout)

    for col in HR_ZONE_COLS + ["icu_training_load", "icu_fitness", "icu_fatigue", "average_heartrate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("start_date_local").reset_index(drop=True)
