"""Small display-formatting helpers shared across the app.

The aggregation logic that used to live here (weekly summary, personal
bests, HR zones, easy-effort cohort, summary stats) now lives in
dbt/models/ and is queried via src/warehouse.py -- it's genuinely SQL-shaped
work, which is what dbt/DuckDB are for.
"""
import pandas as pd


def format_pace(pace_min_per_km: float) -> str:
    if pd.isna(pace_min_per_km):
        return "-"
    minutes = int(pace_min_per_km)
    seconds = round((pace_min_per_km - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d} /km"
