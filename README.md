# Running Dashboard

An interactive Streamlit dashboard analyzing my running data exported from
[intervals.icu](https://intervals.icu): weekly volume, fitness/fatigue/form trend,
heart-rate zone distribution, personal bests, and a full run log.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Data

`data/activities.csv` is a raw intervals.icu activity export. It includes heart-rate,
pace, and weight data — before making this repo public, decide whether you're
comfortable with that being visible, or strip/aggregate those columns first.

## Deploying the dashboard

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with the same
   GitHub account, and deploy from this repo, with `app.py` as the entry point.
3. Streamlit Cloud installs `requirements.txt` automatically — no extra config needed.

## Roadmap

- [ ] AI-generated training suggestions based on training load / HR zone balance
- [ ] Weekly mileage progression vs. injury-risk heuristics (e.g. acute:chronic workload ratio)
