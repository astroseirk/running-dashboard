"""Race-time prediction via ML models trained on individual training runs.

Reframes the problem from "3 races = 3 samples" (too small for real ML) to
"every run is a training row": features describe the training context
leading up to that run (rolling volume, hard/easy-effort km, prior fitness,
running history) and the target is that run's own time. This gives ~40-170
rows per race, fit only on runs strictly before that race's date (never the
race itself), which supports real cross-validation and model comparison.

Key finding baked into this design: general cross-validation error (dominated
by the many short/easy training runs) does NOT predict how well a model
extrapolates to the one rare, long, race-distance effort we actually care
about. Tree/kernel methods (Random Forest, RBF-SVM) look good on ordinary
runs but badly under-predict race time because they average toward the bulk
of the (mostly short) training distribution. A "long-run holdout" check
(train without the longest few runs, test on them) catches this and should
be trusted over plain CV MAE when picking a model here.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.race_prediction import RACES

FEATURES = [
    "log_distance_km",
    "roll7_km",
    "roll28_km",
    "roll28_hard_km",
    "roll28_easy_km",
    "roll28_n_runs",
    "long_run_28d_km",
    "days_running_history",
    "prior_fitness",
    "prior_fatigue",
]

FEATURE_GROUPS = {
    "Volume": ["roll7_km", "roll28_km", "roll28_n_runs", "long_run_28d_km"],
    "Hard effort": ["roll28_hard_km"],
    "Easy effort": ["roll28_easy_km"],
    "Fitness/history": ["days_running_history", "prior_fitness", "prior_fatigue"],
}

HARD_EFFORT_TYPES = ("Tempo", "Intervals", "Race")
HARD_EFFORT_RPE_MIN = 6
EASY_EFFORT_RPE_MAX = 3


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("start_date_local").reset_index(drop=True).copy()
    idx = pd.DatetimeIndex(df["start_date_local"])
    dist = pd.Series(df["distance_km"].values, index=idx)

    hard_mask = df["workout_type"].isin(HARD_EFFORT_TYPES) | (df["icu_rpe"] >= HARD_EFFORT_RPE_MIN)
    easy_mask = df["icu_rpe"] <= EASY_EFFORT_RPE_MAX
    dist_hard = dist.where(hard_mask.values, 0.0)
    dist_easy = dist.where(easy_mask.values, 0.0)
    ones = pd.Series(1.0, index=idx)

    # closed="left": each rolling window looks only at runs strictly before the
    # current row's timestamp, so nothing here can leak same-day/future data.
    df["roll7_km"] = dist.rolling("7D", closed="left").sum().values
    df["roll28_km"] = dist.rolling("28D", closed="left").sum().values
    df["roll28_hard_km"] = dist_hard.rolling("28D", closed="left").sum().values
    df["roll28_easy_km"] = dist_easy.rolling("28D", closed="left").sum().values
    df["roll28_n_runs"] = ones.rolling("28D", closed="left").sum().values
    df["long_run_28d_km"] = dist.rolling("28D", closed="left").max().values
    df["days_running_history"] = (df["start_date_local"] - df["start_date_local"].min()).dt.days
    df["prior_fitness"] = df["icu_fitness"].shift(1)
    df["prior_fatigue"] = df["icu_fatigue"].shift(1)
    df["log_distance_km"] = np.log(df["distance_km"])
    df["log_time"] = np.log(df["moving_time"])
    return df


def _model_candidates() -> dict:
    return {
        "Linear Regression": [
            (Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), {}),
        ],
        "SVM (RBF)": [
            (
                Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=C, epsilon=eps))]),
                {"C": C, "epsilon": eps},
            )
            for C in [0.1, 1, 10]
            for eps in [0.01, 0.05]
        ],
        "SVM (linear)": [
            (
                Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear", C=C, epsilon=eps))]),
                {"C": C, "epsilon": eps},
            )
            for C in [0.1, 1, 10]
            for eps in [0.01, 0.05]
        ],
        "Random Forest": [
            (
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            RandomForestRegressor(n_estimators=300, max_depth=depth, min_samples_leaf=leaf, random_state=42),
                        ),
                    ]
                ),
                {"max_depth": depth, "min_samples_leaf": leaf},
            )
            for depth in [3, 5, None]
            for leaf in [1, 3, 5]
        ],
    }


def _cv_mae_minutes(pipe, X, y_log, n_splits) -> float:
    cv = TimeSeriesSplit(n_splits=n_splits)
    preds, actuals = [], []
    for train_idx, test_idx in cv.split(X):
        model = clone(pipe)
        model.fit(X[train_idx], y_log[train_idx])
        preds.extend(model.predict(X[test_idx]))
        actuals.extend(y_log[test_idx])
    pred_min = np.exp(np.array(preds)) / 60
    actual_min = np.exp(np.array(actuals)) / 60
    return mean_absolute_error(actual_min, pred_min)


def ml_race_report(df: pd.DataFrame, race_label: str) -> dict:
    feat = build_features(df)
    race_date = RACES[race_label]["date"]
    cutoff = pd.Timestamp(race_date)

    train = feat[feat["start_date_local"] < cutoff].dropna(subset=FEATURES + ["log_time"])
    race_row = feat[feat["start_date_local"].dt.date == cutoff.date()]
    result = {"race_label": race_label, "n_training_runs": len(train)}
    if race_row.empty or len(train) < 20:
        result["ok"] = False
        return result

    X = train[FEATURES].values
    y_log = train["log_time"].values
    n_splits = 5 if len(train) >= 50 else 3
    actual_min = race_row["moving_time"].values[0] / 60
    x_race = race_row[FEATURES].values

    n_holdout = min(5, max(2, len(train) // 15))
    order = np.argsort(-train["distance_km"].values)
    holdout_idx, keep_idx = order[:n_holdout], order[n_holdout:]

    models = {}
    for model_name, candidates in _model_candidates().items():
        best_mae, best_params, best_pipe = np.inf, None, None
        for pipe, params in candidates:
            mae = _cv_mae_minutes(pipe, X, y_log, n_splits)
            if mae < best_mae:
                best_mae, best_params, best_pipe = mae, params, pipe

        lr_model = clone(best_pipe).fit(X[keep_idx], y_log[keep_idx])
        long_run_pred = np.exp(lr_model.predict(X[holdout_idx])) / 60
        long_run_actual = np.exp(y_log[holdout_idx]) / 60
        long_run_mae = mean_absolute_error(long_run_actual, long_run_pred)

        final_model = clone(best_pipe).fit(X, y_log)
        pred_min = np.exp(final_model.predict(x_race)[0]) / 60

        models[model_name] = {
            "cv_mae": best_mae,
            "long_run_holdout_mae": long_run_mae,
            "predicted_min": pred_min,
            "error_pct": 100 * (pred_min - actual_min) / actual_min,
            "params": best_params,
            "fitted_model": final_model,
        }

    # Standardized linear-regression coefficients, grouped, for the "what actually
    # drives the prediction" breakdown (trusted over Random Forest importances,
    # since RF is one of the models this same analysis shows extrapolates badly here).
    scaler = StandardScaler().fit(X)
    lr = LinearRegression().fit(scaler.transform(X), y_log)
    coefs = dict(zip(FEATURES, lr.coef_))
    grouped_coefs = {
        group: sum(abs(coefs[c]) for c in cols if c in coefs) for group, cols in FEATURE_GROUPS.items()
    }

    result.update(
        {
            "ok": True,
            "actual_min": actual_min,
            "distance_km": race_row["distance_km"].values[0],
            "n_holdout": n_holdout,
            "models": models,
            "coefficients": coefs,
            "grouped_coefficients": grouped_coefs,
        }
    )
    return result
