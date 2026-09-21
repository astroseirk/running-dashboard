import plotly.express as px
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_dark"

from src.data import load_activities
from src.metrics import (
    fitness_fatigue_trend,
    format_pace,
    hr_zone_totals,
    personal_bests,
    summary_stats,
    weekly_summary,
)

st.set_page_config(page_title="Running Dashboard", page_icon="🏃", layout="wide")

df = load_activities()

st.title("🏃 Running Dashboard")
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

tab_trends, tab_zones, tab_bests, tab_table = st.tabs(
    ["Trends", "HR Zones", "Personal Bests", "All Runs"]
)

with tab_trends:
    st.subheader("Weekly distance")
    weekly = weekly_summary(df)
    fig = px.bar(weekly, x="week", y="distance_km", labels={"week": "Week", "distance_km": "Distance (km)"})
    st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Fitness/fatigue from intervals.icu's training-load model. Form = fitness − fatigue.")

    st.subheader("Pace over time")
    fig3 = px.scatter(
        df,
        x="start_date_local",
        y="pace_min_per_km",
        color="workout_type",
        labels={"start_date_local": "Date", "pace_min_per_km": "Pace (min/km)"},
    )
    fig3.update_yaxes(autorange="reversed")  # faster pace (lower number) reads as "up"
    st.plotly_chart(fig3, use_container_width=True)

with tab_zones:
    st.subheader("Time in heart-rate zone by month")
    zones = hr_zone_totals(df, by="month")
    zone_cols = [c for c in zones.columns if c.startswith("Zone")]
    if zone_cols:
        fig4 = px.bar(zones, x="month", y=zone_cols, labels={"month": "Month", "value": "Hours"})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No heart-rate zone data available.")

with tab_bests:
    st.subheader("Best efforts by distance")
    bests = personal_bests(df)
    if bests.empty:
        st.info("No runs matched the standard race distances (5K / 10K / half marathon) yet.")
    else:
        st.dataframe(bests, use_container_width=True, hide_index=True)

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
    st.dataframe(display.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
