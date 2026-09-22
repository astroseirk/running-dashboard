"""Query layer over the dbt-built DuckDB warehouse (dbt/running.duckdb).

Powers every BI/aggregation tab in the dashboard -- these are genuinely
SQL-shaped (group-by, window functions, ranking), which is exactly what the
dbt staging/mart layer in dbt/models/ is for. Every function here queries
`stg_activities` (the dbt staging model) with the sidebar's live filters
applied as SQL WHERE clauses, so the dbt marts and this module encode the
*same* transformation logic -- the marts are the fixed, unfiltered case of
the same query.

Deliberately NOT used by the race-prediction ML path (src/race_ml.py,
src/race_prediction.py): that needs per-run causal rolling-window features
fed into scikit-learn pipelines, which is a better fit for pandas than SQL,
and it needs the full unfiltered history regardless of sidebar state anyway.
That path still reads data/activities.csv directly via src/data.py. Both
paths trace back to the same canonical export -- dbt/generate_seed.py
regenerates the dbt seed from it.
"""
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "dbt" / "running.duckdb"


def _connect():
    return duckdb.connect(str(DB_PATH), read_only=True)


def date_bounds() -> tuple:
    with _connect() as con:
        row = con.sql("select min(start_date_local), max(start_date_local) from stg_activities").fetchone()
    return row[0].date(), row[1].date()


def distinct_workout_types() -> list:
    with _connect() as con:
        return sorted(r[0] for r in con.sql("select distinct workout_type from stg_activities").fetchall())


def _run_query(sql: str, params: list) -> pd.DataFrame:
    with _connect() as con:
        return con.execute(sql, params).df()


def get_filtered_activities(start_date, end_date, workout_types: list) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(workout_types))
    sql = f"""
        select *
        from stg_activities
        where cast(start_date_local as date) between ? and ?
          and workout_type in ({placeholders})
        order by start_date_local
    """
    params = [start_date, end_date, *workout_types]
    return _run_query(sql, params)


def summary_stats(start_date, end_date, workout_types: list) -> dict:
    placeholders = ",".join(["?"] * len(workout_types))
    sql = f"""
        select
            count(*) as total_runs,
            sum(distance_km) as total_distance_km,
            sum(moving_time_sec) / 3600.0 as total_time_hours,
            avg(pace_min_per_km) as avg_pace_min_per_km
        from stg_activities
        where cast(start_date_local as date) between ? and ?
          and workout_type in ({placeholders})
    """
    row = _run_query(sql, [start_date, end_date, *workout_types]).iloc[0]
    return {
        "total_runs": int(row["total_runs"]),
        "total_distance_km": row["total_distance_km"] or 0.0,
        "total_time_hours": row["total_time_hours"] or 0.0,
        "avg_pace_min_per_km": row["avg_pace_min_per_km"],
    }


def weekly_summary(start_date, end_date, workout_types: list) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(workout_types))
    sql = f"""
        select week, sum(distance_km) as distance_km, count(*) as runs,
               sum(icu_training_load) as training_load, avg(pace_min_per_km) as avg_pace
        from stg_activities
        where cast(start_date_local as date) between ? and ?
          and workout_type in ({placeholders})
        group by week
        order by week
    """
    return _run_query(sql, [start_date, end_date, *workout_types])


def hr_zone_totals(start_date, end_date, workout_types: list) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(workout_types))
    sql = f"""
        select
            month,
            sum(hr_z1_secs) / 3600.0 as "Zone 1",
            sum(hr_z2_secs) / 3600.0 as "Zone 2",
            sum(hr_z3_secs) / 3600.0 as "Zone 3",
            sum(hr_z4_secs) / 3600.0 as "Zone 4",
            sum(hr_z5_secs) / 3600.0 as "Zone 5",
            sum(hr_z6_secs) / 3600.0 as "Zone 6",
            sum(hr_z7_secs) / 3600.0 as "Zone 7"
        from stg_activities
        where cast(start_date_local as date) between ? and ?
          and workout_type in ({placeholders})
        group by month
        order by month
    """
    return _run_query(sql, [start_date, end_date, *workout_types])


def personal_bests(start_date, end_date, workout_types: list) -> pd.DataFrame:
    from src.metrics import format_pace

    placeholders = ",".join(["?"] * len(workout_types))
    sql = f"""
        with in_range as (
            select * from stg_activities
            where cast(start_date_local as date) between ? and ?
              and workout_type in ({placeholders})
        ),
        bucketed as (
            select *,
                case
                    when distance_km between 4.8 and 5.3 then '5K'
                    when distance_km between 9.7 and 10.3 then '10K'
                    when distance_km between 20.5 and 21.6 then 'Half Marathon'
                    when distance_km between 41.5 and 43.5 then 'Marathon'
                end as pb_distance
            from in_range
            where pace_min_per_km is not null
        ),
        ranked as (
            select *, row_number() over (partition by pb_distance order by pace_min_per_km asc) as rn
            from bucketed
            where pb_distance is not null
        )
        select
            pb_distance as "Distance",
            cast(start_date_local as date) as "Date",
            name as "Run",
            pace_min_per_km,
            moving_time_min as "Time (min)"
        from ranked
        where rn = 1
        order by case pb_distance when '5K' then 1 when '10K' then 2 when 'Half Marathon' then 3 when 'Marathon' then 4 end
    """
    df = _run_query(sql, [start_date, end_date, *workout_types])
    if not df.empty:
        df["Pace"] = df["pace_min_per_km"].apply(format_pace)
        df = df.drop(columns="pace_min_per_km")[["Distance", "Date", "Run", "Pace", "Time (min)"]]
    return df


EASY_EFFORT_RPE_MAX = 3


def easy_effort_monthly(start_date, end_date) -> pd.DataFrame:
    sql = """
        with easy as (
            select *, 60.0 / pace_min_per_km as speed_kmh
            from stg_activities
            where cast(start_date_local as date) between ? and ?
              and icu_rpe <= ?
              and average_heartrate is not null
              and pace_min_per_km is not null
        )
        select
            month as start_date_local,
            avg(average_heartrate) as avg_hr,
            avg(pace_min_per_km) as avg_pace,
            avg(speed_kmh / average_heartrate) as efficiency,
            count(*) as runs
        from easy
        group by month
        order by month
    """
    return _run_query(sql, [start_date, end_date, EASY_EFFORT_RPE_MAX])


def easy_effort_half_split(start_date, end_date):
    sql = """
        with easy as (
            select *, 60.0 / pace_min_per_km as speed_kmh
            from stg_activities
            where cast(start_date_local as date) between ? and ?
              and icu_rpe <= ?
              and average_heartrate is not null
              and pace_min_per_km is not null
        ),
        with_eff as (
            select *, speed_kmh / average_heartrate as efficiency from easy
        ),
        halved as (
            select *, ntile(2) over (order by start_date_local) as half from with_eff
        )
        select
            half,
            count(*) as n,
            avg(average_heartrate) as avg_hr,
            avg(pace_min_per_km) as avg_pace,
            avg(efficiency) as efficiency
        from halved
        group by half
        order by half
    """
    df = _run_query(sql, [start_date, end_date, EASY_EFFORT_RPE_MAX])
    if len(df) < 2 or df["n"].sum() < 4:
        return None
    first, second = df.iloc[0], df.iloc[1]
    return {
        "first": {"n": int(first["n"]), "avg_hr": first["avg_hr"], "avg_pace": first["avg_pace"], "efficiency": first["efficiency"]},
        "second": {"n": int(second["n"]), "avg_hr": second["avg_hr"], "avg_pace": second["avg_pace"], "efficiency": second["efficiency"]},
    }
