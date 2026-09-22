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
  the fit — genuine out-of-sample prediction, not curve-fitting through a handful of points. A second
  section compares Linear Regression, SVM (RBF + linear kernel), and Random Forest on the same
  out-of-sample task — every training run becomes a feature row (rolling volume, hard/easy-effort km,
  running history, prior fitness/fatigue), enough data for real cross-validation. It also surfaces a
  genuine, non-obvious finding: tree/kernel methods look best on ordinary cross-validation but badly
  under-predict the actual race, because they can't extrapolate past the bulk of the (mostly short)
  training distribution — a "long-run holdout" check catches this and is what the recommendation is
  based on, not plain CV score. Method and results are explained in-app per race.
- **Personal Bests** — best pace at 5K / 10K / half marathon / marathon distances
- **All Runs** — full sortable run log

Sidebar filters (date range, workout type) apply across tabs, except Race Prediction, which always
uses full history regardless of filters (the method needs everything before each race date).

## Data

`data/activities.csv` is a raw intervals.icu activity export (running activities only), including
heart-rate and pace data. This repo and its data are public by choice.

## Architecture: dbt + DuckDB warehouse

Every BI/aggregation tab (Trends, HR Zones, Easy Effort Trend, Personal Bests, All Runs) is backed
by a real dbt project (`dbt/`) on DuckDB, not ad-hoc pandas munging:

```
data/activities.csv  --generate_seed.py-->  dbt/seeds/activities.csv  --dbt seed/run-->  dbt/running.duckdb
                                                                                              |
                                                                          staging: stg_activities (typed, cleaned, classified)
                                                                                              |
                                                       marts: mart_weekly_summary, mart_easy_effort_monthly,
                                                              mart_personal_bests, mart_hr_zone_totals
```

`src/warehouse.py` queries `dbt/running.duckdb` at request time (parameterized by the sidebar's live
date/workout-type filters), so the dashboard and the fixed dbt marts encode the *same* SQL logic —
the marts are just the unfiltered special case. 15 dbt tests (not-null, unique, accepted-values)
pass on every model. This isn't decorative: it replaced what used to be pandas aggregation code in
`src/metrics.py`.

**Deliberately not dbt:** Race Prediction's feature engineering (causal rolling windows, log
transforms) and scikit-learn model fitting stay in pandas — that's genuinely a better fit for
per-run ML features than SQL, and it needs the full unfiltered history regardless of sidebar state.
That path reads `data/activities.csv` directly. Both paths trace back to the same source file.

To rebuild the warehouse after updating `data/activities.csv`:

```bash
pip install -r dbt/requirements-dbt.txt
python dbt/generate_seed.py
cd dbt
dbt seed --profiles-dir . && dbt run --profiles-dir . && dbt test --profiles-dir .
```

`dbt/running.duckdb` is committed (Streamlit Cloud doesn't run a dbt build step), so remember to
commit the rebuilt file alongside `dbt/seeds/activities.csv`.

**Next up:** a Databricks Community Edition notebook reproducing the feature engineering / model
comparison in PySpark, as a documented artifact (not live-wired into this app — a free personal
project has no business holding standing credentials to a production-grade Databricks workspace).

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                     Streamlit UI — all tabs
src/warehouse.py            Queries dbt/running.duckdb for every BI tab
src/data.py                 Loads + cleans the intervals.icu export (ML path only)
src/metrics.py               format_pace() display helper
src/race_prediction.py       Riegel power-law fit + out-of-sample race prediction
src/race_ml.py                Linear Regression / SVM / Random Forest comparison
dbt/                        dbt project: seeds, staging, marts, tests (see above)
data/activities.csv          Raw intervals.icu export (canonical source)
.streamlit/config.toml      Dark theme
```

## Deploying

Already deployed via [Streamlit Community Cloud](https://share.streamlit.io), which auto-redeploys
on every push to `main`. To redeploy elsewhere: push this repo to GitHub, deploy from
share.streamlit.io with `app.py` as the entry point — `requirements.txt` is installed automatically
(note: this installs `duckdb` to *query* the warehouse, not `dbt-core` to *build* it — the build step
runs locally, see above).

## Roadmap

- [ ] Databricks Community Edition notebook: PySpark version of the feature engineering / model comparison
- [ ] Extend Race Prediction to a future/in-progress training block (not just backtesting past races)
- [ ] Weekly mileage progression vs. injury-risk heuristics (e.g. acute:chronic workload ratio)
