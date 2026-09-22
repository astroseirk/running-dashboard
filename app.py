import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_dark"

from src import warehouse as wh
from src.data import load_activities
from src.metrics import format_pace
from src.race_ml import ml_race_report
from src.race_prediction import RACES, race_report

st.set_page_config(page_title="Running Dashboard", page_icon="🏃", layout="wide")


@st.cache_data(show_spinner="Fitting Linear Regression / SVM / Random Forest (grid search + cross-validation)...")
def _cached_ml_race_report(df, race_choice):
    return ml_race_report(df, race_choice)


# Race prediction needs the full unfiltered history as a pandas DataFrame for its
# per-run rolling-window feature engineering, regardless of the sidebar filters below.
df_all = load_activities()

st.title("🏃 Running Dashboard")
st.markdown(
    "A personal project analyzing **my own running data** — every run here is real, tracked via "
    "Garmin and synced to [intervals.icu](https://intervals.icu) since January 2024."
)
st.caption(
    f"{len(df_all)} runs loaded — most tabs query a dbt-built DuckDB warehouse "
    "(`dbt/models/`); Race Prediction uses pandas/scikit-learn directly on the raw export."
)

# --- Sidebar filters ---
st.sidebar.header("Filters")
min_date, max_date = wh.date_bounds()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
all_workout_types = wh.distinct_workout_types()
workout_types = st.sidebar.multiselect("Workout type", options=all_workout_types, default=all_workout_types)

start, end = date_range if len(date_range) == 2 else (min_date, max_date)

if not workout_types:
    st.warning("No runs match the current filters.")
    st.stop()

filtered = wh.get_filtered_activities(start, end, workout_types)
if filtered.empty:
    st.warning("No runs match the current filters.")
    st.stop()

# --- Overview ---
stats = wh.summary_stats(start, end, workout_types)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runs", stats["total_runs"])
col2.metric("Total distance", f"{stats['total_distance_km']:.0f} km")
col3.metric("Total time", f"{stats['total_time_hours']:.0f} h")
col4.metric("Average pace", format_pace(stats["avg_pace_min_per_km"]))

tab_trends, tab_zones, tab_easy, tab_predict, tab_bests, tab_table = st.tabs(
    ["Trends", "HR Zones", "Easy Effort Trend", "Race Prediction", "Personal Bests", "All Runs"]
)

with tab_trends:
    st.subheader("Weekly distance")
    weekly = wh.weekly_summary(start, end, workout_types)
    fig = px.bar(weekly, x="week", y="distance_km", labels={"week": "Week", "distance_km": "Distance (km)"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Fitness, fatigue & form")
    trend = filtered[["start_date_local", "icu_fitness", "icu_fatigue"]].dropna()
    if trend.empty:
        st.info("No fitness/fatigue data in the selected range.")
    else:
        trend = trend.copy()
        trend["form"] = trend["icu_fitness"] - trend["icu_fatigue"]
        fig2 = px.line(
            trend,
            x="start_date_local",
            y=["icu_fitness", "icu_fatigue", "form"],
            labels={"start_date_local": "Date", "value": "Score", "variable": "Metric"},
        )
        st.plotly_chart(fig2, width='stretch')
        st.caption("Fitness/fatigue from intervals.icu's training-load model. Form = fitness − fatigue.")

    st.subheader("Pace over time")
    fig3 = px.scatter(
        filtered,
        x="start_date_local",
        y="pace_min_per_km",
        color="workout_type",
        labels={"start_date_local": "Date", "pace_min_per_km": "Pace (min/km)"},
    )
    fig3.update_yaxes(autorange="reversed")  # faster pace (lower number) reads as "up"
    st.plotly_chart(fig3, width='stretch')

with tab_zones:
    st.subheader("Time in heart-rate zone by month")
    zones = wh.hr_zone_totals(start, end, workout_types)
    zone_cols = [c for c in zones.columns if c.startswith("Zone")]
    if zone_cols:
        fig4 = px.bar(zones, x="month", y=zone_cols, labels={"month": "Month", "value": "Hours"})
        st.plotly_chart(fig4, width='stretch')
    else:
        st.info("No heart-rate zone data available.")

with tab_easy:
    st.subheader("Easy-effort HR & pace trend")
    st.caption(
        "Cohort is every run with RPE ≤ 3, not just runs titled 'Easy Run' — "
        "the name tag only exists on a small, recent subset and understates how many easy runs you've done."
    )
    monthly = wh.easy_effort_monthly(start, end)
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
        st.caption("Efficiency = speed ÷ average HR. Rising over time means more speed for the same effort — an aerobic-fitness signal on easy days specifically.")

        split = wh.easy_effort_half_split(start, end)
        if split:
            st.markdown("**First half vs. second half of this date range**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg HR", f"{split['second']['avg_hr']:.0f} bpm", f"{split['second']['avg_hr'] - split['first']['avg_hr']:+.1f} bpm")
            c2.metric("Avg pace", format_pace(split["second"]["avg_pace"]), f"{split['second']['avg_pace'] - split['first']['avg_pace']:+.2f} min/km")
            c3.metric("Efficiency", f"{split['second']['efficiency']:.4f}", f"{split['second']['efficiency'] - split['first']['efficiency']:+.4f}")

with tab_predict:
    st.subheader("Race time prediction")
    st.caption(
        "For the selected race, every training run strictly BEFORE that race date is used to fit a "
        "prediction — the race's own result is never part of the fit."
    )
    race_choice = st.selectbox("Race", list(RACES.keys()))
    result = race_report(df_all, race_choice)

    if not result.get("found"):
        st.warning("Could not find this race in the data.")
    elif not result["fit_ok"]:
        st.warning("Not enough hard-effort training runs before this race to fit a prediction.")
    else:
        st.markdown(
            f"""
**Method:** Riegel's power law, the standard race-time-scaling formula from exercise
science: `time = a × distance^b`. `a` and `b` are fit (log-log linear regression) using
every **hard-effort** run before {result['race_date']} — anything tagged Tempo, Intervals,
or Race, or self-rated RPE ≥ 6. Easy/recovery runs are excluded because they don't represent
a maximal sustainable pace for their duration, which is what the formula needs to extrapolate
from.
"""
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training runs before race", result["n_training_runs"])
        c2.metric("Hard-effort points used in fit", len(result["hard_effort_points"]))
        c3.metric("Fitted exponent (b)", f"{result['b']:.3f}")
        c4.metric(
            "Fit duration range",
            f"{result['min_duration_min']:.0f}–{result['max_duration_min']:.0f} min",
        )

        pred_min = result["predicted_seconds"] / 60
        actual_min = result["actual_seconds"] / 60
        d1, d2, d3 = st.columns(3)
        d1.metric("Predicted", f"{format_pace(pred_min / result['distance_km'])}", f"{pred_min:.1f} min total")
        d2.metric("Actual", f"{format_pace(actual_min / result['distance_km'])}", f"{actual_min:.1f} min total")
        d3.metric(
            "Prediction error",
            f"{result['error_seconds']/60:+.1f} min",
            f"{result['error_pct']:+.1f}%",
            delta_color="inverse",
        )

        # Fit curve + points, on a log-log duration-vs-distance plot
        hard = result["hard_effort_points"]
        fig8 = go.Figure()
        fig8.add_trace(
            go.Scatter(
                x=hard["distance"] / 1000,
                y=hard["moving_time"] / 60,
                mode="markers",
                name="Hard-effort training runs (used in fit)",
                text=hard["name"],
                marker=dict(size=8),
            )
        )
        x_curve = np.linspace(hard["distance"].min(), max(hard["distance"].max(), result["distance_km"] * 1000), 200)
        y_curve = result["a"] * x_curve**result["b"] / 60
        fig8.add_trace(go.Scatter(x=x_curve / 1000, y=y_curve, mode="lines", name="Fitted curve (extrapolated)"))
        fig8.add_trace(
            go.Scatter(
                x=[result["distance_km"]],
                y=[pred_min],
                mode="markers",
                name="Predicted",
                marker=dict(size=14, symbol="diamond"),
            )
        )
        fig8.add_trace(
            go.Scatter(
                x=[result["distance_km"]],
                y=[actual_min],
                mode="markers",
                name="Actual",
                marker=dict(size=14, symbol="star"),
            )
        )
        fig8.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title="Duration (min)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig8, width='stretch')

        if result["error_pct"] > 0:
            st.caption(
                f"You ran {abs(result['error_pct']):.1f}% faster than the model predicted from your training "
                "efforts alone — a real, positive signal (race-day execution, taper, or conservative training "
                "paces), not a modeling flaw."
            )
        else:
            st.caption(
                f"You ran {abs(result['error_pct']):.1f}% slower than the model predicted from your training "
                "efforts — could be race-day conditions, pacing, or fueling; worth a look."
            )

        st.divider()
        st.subheader("Model comparison: Linear Regression vs. SVM vs. Random Forest")
        st.markdown(
            """
**Method:** every individual training run before this race becomes one training row —
features describe the training context at that point (7-/28-day rolling volume, rolling
hard-effort km, rolling easy-effort km, run frequency, longest recent run, running history,
prior fitness/fatigue from intervals.icu's own model), and the target is that run's own
time. This gives far more rows than the 3-race version above, enough for real
cross-validation. All features are standardized (zero mean, unit variance) before fitting.

**Why two error columns:** plain cross-validation MAE is measured on ordinary training
runs, which are mostly short. It does **not** tell you how a model will extrapolate to the
one long, rare, race-distance effort you actually care about — a model can look great on
CV and still fail badly at the actual race. *Long-run holdout MAE* (train without the
longest few runs, test on exactly those) is a much better proxy for that, and is what the
recommendation below is based on.
"""
        )

        ml_result = _cached_ml_race_report(df_all, race_choice)
        if not ml_result.get("ok"):
            st.warning("Not enough training runs before this race to fit ML models.")
        else:
            rows = []
            for name, m in ml_result["models"].items():
                rows.append(
                    {
                        "Model": name,
                        "CV MAE (min)": round(m["cv_mae"], 1),
                        "Long-run holdout MAE (min)": round(m["long_run_holdout_mae"], 1),
                        "Race prediction (min)": round(m["predicted_min"], 1),
                        "Error": f"{m['error_pct']:+.1f}%",
                    }
                )
            model_df = pd.DataFrame(rows).sort_values("Long-run holdout MAE (min)")
            best_name = model_df.iloc[0]["Model"]
            st.dataframe(model_df, width="stretch", hide_index=True)
            st.caption(
                f"Actual: {ml_result['actual_min']:.1f} min. Lowest long-run-holdout error here: **{best_name}** — "
                "that's the one I'd trust for this race, even if it isn't the lowest plain-CV-MAE model above."
            )

            st.markdown("**Feature value: how much does each factor matter?**")
            st.caption(
                "Standardized linear-regression coefficients, grouped and summed (absolute value) — "
                "excludes distance itself, which trivially dominates (of course a marathon takes longer "
                "than a 5K) and would drown out everything else."
            )
            coef_df = pd.DataFrame(
                {"Factor": list(ml_result["grouped_coefficients"].keys()), "Relative weight": list(ml_result["grouped_coefficients"].values())}
            ).sort_values("Relative weight", ascending=True)
            fig9 = px.bar(coef_df, x="Relative weight", y="Factor", orientation="h")
            st.plotly_chart(fig9, width="stretch")

with tab_bests:
    st.subheader("Best efforts by distance")
    bests = wh.personal_bests(start, end, workout_types)
    if bests.empty:
        st.info("No runs matched the standard race distances (5K / 10K / half marathon) yet.")
    else:
        st.dataframe(bests, width='stretch', hide_index=True)

with tab_table:
    st.subheader("All runs")
    display = filtered[
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
