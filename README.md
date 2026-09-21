# Running Dashboard

**Live app:** https://running-dashboard-9xckg9wyw4isfx4dney6rp.streamlit.app/

An interactive Streamlit dashboard analyzing my running data exported from
[intervals.icu](https://intervals.icu).

## About

This is a personal side project built on **my own running data** — every activity here is a real
run I did, tracked via Garmin and synced to intervals.icu, going back to January 2024. I built this
to dig into my own training patterns (pace, heart rate, training load) beyond what a stock app
shows, and to try something more ambitious: predicting my race times from training data alone, and
checking those predictions against races I've actually run. It's a Python/Streamlit project, not
affiliated with any employer.

## What's in it

- **Trends** — weekly distance, fitness/fatigue/form (intervals.icu's training-load model), pace over time
- **HR Zones** — time spent in each heart-rate zone, by month
- **Easy Effort Trend** — HR/pace/aerobic-efficiency trend on easy-effort runs (RPE ≤ 3), not just
  runs literally titled "Easy Run" — that tag only exists on a small, recent subset
- **Race Prediction** — for each of 3 selected races, fits Riegel's power law
  (`time = a × distance^b`) on hard-effort training runs strictly *before* that race date, then
  predicts the race and compares to what actually happened. The race's own result is never part of
  the fit — genuine out-of-sample prediction, not curve-fitting through a handful of points. Method
  and results are explained in-app per race.
- **Personal Bests** — best pace at 5K / 10K / half marathon / marathon distances
- **All Runs** — full sortable run log

Sidebar filters (date range, workout type) apply across tabs, except Race Prediction, which always
uses full history regardless of filters (the method needs everything before each race date).

## Data

`data/activities.csv` is a raw intervals.icu activity export (running activities only), including
heart-rate and pace data. This repo and its data are public by choice.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                  Streamlit UI — all tabs
src/data.py              Loads + cleans the intervals.icu export
src/metrics.py            Weekly volume, fitness/fatigue trend, HR zones, personal bests, easy-effort cohort
src/race_prediction.py    Riegel power-law fit + out-of-sample race prediction
data/activities.csv       Raw intervals.icu export
.streamlit/config.toml    Dark theme
```

## Deploying

Already deployed via [Streamlit Community Cloud](https://share.streamlit.io), which auto-redeploys
on every push to `main`. To redeploy elsewhere: push this repo to GitHub, deploy from
share.streamlit.io with `app.py` as the entry point — `requirements.txt` is installed automatically.

## Roadmap

- [ ] Extend Race Prediction to a future/in-progress training block (not just backtesting past races)
- [ ] Weekly mileage progression vs. injury-risk heuristics (e.g. acute:chronic workload ratio)
