import plotly.express as px
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_dark"

from src.data import load_activities
from src.metrics import (
    easy_effort_half_split,
    easy_effort_monthly,
    fitness_fatigue_trend,
    format_pace,
    hr_zone_totals,
    personal_bests,
    summary_stats,
    weekly_summary,
)

st.set_page_config(page_title="Running Dashboard", page_icon="ðŸƒ", layout="wide")

df = load_activities()

st.title("ðŸƒ Running Dashboard")
st.caption(f"{len(df)} runs loaded from intervals.icu export")

# --- Sidebar filters ---
st.sidebar.header("Filters")
min_date, max_date = df["start_date_local"].min().date(), df["start_date_local"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
workout_types = st.sidebar.multiselect(
    "Workout type", options=sorted(df["workout_type"].unique()), default=sorted(df["workout_type"].unique())
)

if len(date_range) == 2:
    start, end = date_range
    df = df[(df["start_date_local"].dt.date >= start) & (df["start_date_local"].dt.date <= end)]

# Keep a date-filtered-only copy (before the workout-type filter) for the easy-effort
# cohort, which is defined by RPE rather than by the same name-based type filter.
df_dated = df.copy()
df = df[df["workout_type"].isin(workout_types)]

if df.empty:
    st.warning("No runs match the current filters.")
    st.stop()

# --- Overview ---
stats = summary_stats(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runs", stats["total_runs"])
col2.metric("Total distance", f"{stats['total_distance_km']:.0f} km")
col3.metric("Total time", f"{stats['total_time_hours']:.0f} h")
col4.metric("Average pace", stats["avg_pace"])

tab_trends, tab_zones, tab_easy, tab_bests, tab_table = st.tabs(
    ["Trends", "HR Zones", "Easy Effort Trend", "Personal Bests", "All Runs"]
)

with tab_trends:
    st.subheader("Weekly distance")
    weekly = weekly_summary(df)
    fig = px.bar(weekly, x="week", y="distance_km", labels={"week": "Week", "distance_km": "Distance (km)"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Fitness, fatigue & form")
    trend = fitness_fatigue_trend(df)
    if trend.empty:
        st.info("No fitness/fatigue data in the selected range.")
    else:
        fig2 = px.line(
            trend,
            x="start_date_local",
            y=["icu_fitness", "icu_fatigue", "form"],
            labels={"start_date_local": "Date", "value": "Score", "variable": "Metric"},
        )
        st.plotly_chart(fig2, width='stretch')
        st.caption("Fitness/fatigue from intervals.icu's training-load model. Form = fitness âˆ’ fatigue.")

    st.subheader("Pace over time")
    fig3 = px.scatter(
        df,
        x="start_date_local",
        y="pace_min_per_km",
        color="workout_type",
        labels={"start_date_local": "Date", "pace_min_per_km": "Pace (min/km)"},
    )
    fig3.update_yaxes(autorange="reversed")  # faster pace (lower number) reads as "up"
    st.plotly_chart(fig3, width='stretch')

with tab_zones:
    st.subheader("Time in heart-rate zone by month")
    zones = hr_zone_totals(df, by="month")
    zone_cols = [c for c in zones.columns if c.startswith("Zone")]
    if zone_cols:
        fig4 = px.bar(zones, x="month", y=zone_cols, labels={"month": "Month", "value": "Hours"})
        st.plotly_chart(fig4, width='stretch')
    else:
        st.info("No heart-rate zone data available.")

with tab_easy:
    st.subheader("Easy-effort HR & pace trend")
    st.caption(
        "Cohort is every run with RPE â‰¤ 3, not just runs titled â€˜Easy Runâ€™ â€” "
        "the name tag only exists on a small, recent subset and understates how many easy runs you've done."
    )
    monthly = easy_effort_monthly(df_dated)
    if monthly.empty:
        st.info("No runs with both an RPE rating and heart-rate data in this date range.")
    else:
        fig5 = px.line(monthly, x="start_date_local", y="avg_hr", markers=True, labels={"start_date_local": "Month", "avg_hr": "Avg HR (bpm)"})
        st.plotly_chart(fig5, width='stretch')

        fig6 = px.line(monthly, x="start_date_local", y="avg_pace", markers=True, labels={"start_date_local": "Month", "avg_pace": "Avg pace (min/km)"})
        fig6.update_yaxes(autorange="reversed")
        st.plotly_chart(fig6, width='stretch')

        fig7 = px.line(
            monthly,
            x="start_date_local",
            y="efficiency",
            markers=True,
            labels={"start_date_local": "Month", "efficiency": "Efficiency (km/h per bpm)"},
        )
        st.plotly_chart(fig7, width='stretch')
        st.caption("Efficiency = speed Ã· average HR. Rising over time means more speed for the same effort â€” an aerobic-fitness signal on easy days specifically.")

        split = easy_effort_half_split(df_dated)
        if split:
            st.markdown("**First half vs. second half of this date range**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg HR", f"{split['second']['avg_hr']:.0f} bpm", f"{split['second']['avg_hr'] - split['first']['avg_hr']:+.1f} bpm")
            c2.metric("Avg pace", format_pace(split["second"]["avg_pace"]), f"{split['second']['avg_pace'] - split['first']['avg_pace']:+.2f} min/km")
            c3.metric("Efficiency", f"{split['second']['efficiency']:.4f}", f"{split['second']['efficiency'] - split['first']['efficiency']:+.4f}")

with tab_bests:
    st.subheader("Best efforts by distance")
    bests = personal_bests(df)
    if bests.empty:
        st.info("No runs matched the standard race distances (5K / 10K / half marathon) yet.")
    else:
        st.dataframe(bests, width='stretch', hide_index=True)

with tab_table:
    st.subheader("All runs")
    display = df[
        ["start_date_local", "name", "workout_type", "distance_km", "moving_time_min", "pace_min_per_km", "average_heartrate"]
    ].copy()
    display["pace"] = display["pace_min_per_km"].apply(format_pace)
    display = display.drop(columns="pace_min_per_km").rename(
        columns={
            "start_date_local": "Date",
            "name": "Name",
            "workout_type": "Type",
            "distance_km": "Distance (km)",
            "moving_time_min": "Time (min)",
            "average_heartrate": "Avg HR",
        }
    )
    st.dataframe(display.sort_values("Date", ascending=False), width='stretch', hide_index=True)
