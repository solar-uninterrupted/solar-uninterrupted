#!/usr/bin/env python3
"""
Solar Uninterrupted — BDA 602 Module 7
Experiment 3: one-year-ahead rooftop-solar adoption-growth forecasting.

Owner: Shashwat Bajaj (Modeling Lead)

Purpose
-------
Predict the next calendar year's increase in postcode-level rooftop-solar
adoption rate:

    annual installations / 2021 occupied private dwellings

This is deliberately different from:

Experiment 1:
    updated-history one-month-ahead installation-count backtest.

Experiment 2:
    fixed-origin end-2015 installation-count forecast.

Experiment 3 is a post-2021 design. It asks whether recent adoption history and
a fixed 2021 Census context help predict the following year's adoption growth.

Chronology
----------
Origin 2021 -> target 2022: tuning-training rows.
Origin 2022 -> target 2023: chronological validation rows.
Origin 2023 -> target 2024: untouched final holdout.

The 2024 holdout is never used for hyperparameter selection or blend weighting.

Feature sets
------------
1. history_only
2. history_plus_census

The Census variables are a fixed 2021 snapshot whose reference date predates
all Experiment 3 target years. Most 2021 Census topics were publicly released
on 28 June 2022. Therefore, the 2021 -> 2022 development row is retrospective
with respect to publication availability, while the 2022 -> 2023 validation
and 2023 -> 2024 holdout are out-of-time forecasts with Census data available
by the forecasting origin. Census variables are never inserted into the
2015-origin Experiment 2.

Primary sample
--------------
- small_pop_flag == 0
- occupied private dwellings >= 50

Sensitivity sample:
- same rules, but occupied private dwellings >= 100

Models
------
- Ridge
- Random Forest
- XGBoost
- validation-selected Random Forest / XGBoost weighted blend

Baselines
---------
- previous year's adoption-rate increment
- mean of previous two annual adoption-rate increments

Evaluation
----------
MAE, RMSE, median absolute error, R².

MLflow
------
The script logs:
- verified M6 Experiment 1 summary
- verified M6 Experiment 2 summary
- M7 Experiment 3 model selections and holdout results

The local mlflow.db SQLite database is gitignored. A compact run index is
saved under results/m7/adoption_growth/mlflow_run_index.csv.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.join_abs import load_joined


OUT = ROOT / "results" / "m7" / "adoption_growth"
MLFLOW_DB = ROOT / "mlflow.db"

POSTCODE = "Small Unit Installation Postcode"
SEED = 42

TRAIN_ORIGIN = 2021
VALID_ORIGIN = 2022
TEST_ORIGIN = 2023
TEST_YEAR = 2024

PRIMARY_MIN_DWELLINGS = 50
SENSITIVITY_MIN_DWELLINGS = 100

HISTORY_FEATURES = [
    "rate_lag1",
    "rate_lag2",
    "rate_lag3",
    "rate_mean2",
    "rate_mean3",
    "rate_trend1",
    "cum_adoption_origin",
    "installations_lag1",
    "installations_mean3",
    "log_dwellings",
    "years_with_activity",
]

CENSUS_NUMERIC = [
    "Median_age_persons",
    "Median_tot_hhd_inc_weekly",
    "Average_household_size",
    "Median_mortgage_repay_monthly",
    "Median_rent_weekly",
    "pop_density_per_sqkm",
    "detached_share",
    "semidetached_share",
    "apartment_share",
    "unoccupied_share",
    "owner_occupied_share",
    "rented_share",
]

CENSUS_CATEGORICAL = ["state_abs"]

FEATURE_SETS = {
    "history_only": {
        "numeric": HISTORY_FEATURES,
        "categorical": [],
    },
    "history_plus_census": {
        "numeric": HISTORY_FEATURES + CENSUS_NUMERIC,
        "categorical": CENSUS_CATEGORICAL,
    },
}


def slug(x: str) -> str:
    return (
        x.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)

    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MedAE": float(median_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "MAE_pct_points": float(100 * mean_absolute_error(y, pred)),
        "RMSE_pct_points": float(100 * np.sqrt(mean_squared_error(y, pred))),
    }


def build_rows(min_dwellings: int) -> pd.DataFrame:
    panel = load_joined(
        ROOT / "data" / "processed" / "cer_abs_panel_long.csv.gz",
        features=ROOT / "data" / "processed" / "abs_poa_features.csv",
    ).copy()

    panel["year"] = panel["month"].dt.year

    static_cols = [
        "dwellings_opd_2021",
        "small_pop_flag",
        "state_abs",
        *CENSUS_NUMERIC,
    ]

    agg = {
        "installations": "sum",
    }
    for c in static_cols:
        agg[c] = "first"

    annual = (
        panel.groupby([POSTCODE, "year"], as_index=False)
        .agg(agg)
        .sort_values([POSTCODE, "year"])
    )

    annual["annual_rate_increment"] = (
        annual["installations"]
        / annual["dwellings_opd_2021"].where(
            annual["dwellings_opd_2021"] > 0
        )
    )

    static = (
        annual.sort_values("year")
        .drop_duplicates(POSTCODE)
        .set_index(POSTCODE)
    )

    eligible = static.index[
        static["small_pop_flag"].eq(0)
        & static["dwellings_opd_2021"].ge(min_dwellings)
    ]

    annual = annual[annual[POSTCODE].isin(eligible)].copy()

    rate = annual.pivot(
        index=POSTCODE,
        columns="year",
        values="annual_rate_increment",
    )

    installs = annual.pivot(
        index=POSTCODE,
        columns="year",
        values="installations",
    ).fillna(0)

    static = static.loc[eligible].copy()

    rows = []

    for origin in [TRAIN_ORIGIN, VALID_ORIGIN, TEST_ORIGIN]:
        target_year = origin + 1

        needed = [origin - 2, origin - 1, origin, target_year]
        missing = [y for y in needed if y not in rate.columns]
        if missing:
            raise RuntimeError(
                f"Missing annual target/history years for origin {origin}: {missing}"
            )

        idx = static.index.intersection(rate.index).intersection(installs.index)

        r = pd.DataFrame(index=idx)
        r["origin"] = origin
        r["target_year"] = target_year

        r["rate_lag1"] = rate.loc[idx, origin]
        r["rate_lag2"] = rate.loc[idx, origin - 1]
        r["rate_lag3"] = rate.loc[idx, origin - 2]
        r["rate_mean2"] = r[["rate_lag1", "rate_lag2"]].mean(axis=1)
        r["rate_mean3"] = r[
            ["rate_lag1", "rate_lag2", "rate_lag3"]
        ].mean(axis=1)
        r["rate_trend1"] = r["rate_lag1"] - r["rate_lag2"]

        hist_years = [y for y in installs.columns if y <= origin]

        cum_installations = installs.loc[idx, hist_years].sum(axis=1)

        dwellings = static.loc[idx, "dwellings_opd_2021"]

        r["cum_adoption_origin"] = (
            cum_installations / dwellings
        )

        r["installations_lag1"] = installs.loc[idx, origin]
        r["installations_mean3"] = (
            installs.loc[idx, [origin - 2, origin - 1, origin]]
            .mean(axis=1)
        )

        r["log_dwellings"] = np.log1p(dwellings)

        r["years_with_activity"] = (
            installs.loc[idx, hist_years].gt(0).sum(axis=1)
        )

        r["target"] = rate.loc[idx, target_year]

        for c in CENSUS_NUMERIC + CENSUS_CATEGORICAL:
            r[c] = static.loc[idx, c]

        r["dwellings_opd_2021"] = dwellings
        r["small_pop_flag"] = static.loc[idx, "small_pop_flag"]

        r = r.reset_index()

        rows.append(r)

    out = pd.concat(rows, ignore_index=True)

    if out["target"].isna().any():
        raise RuntimeError("Unexpected missing target values.")

    return out


def make_preprocessor(numeric: list[str], categorical: list[str]):
    transformers = []

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers.append(("num", numeric_pipe, numeric))

    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ])
        transformers.append(("cat", categorical_pipe, categorical))

    return ColumnTransformer(transformers)


def model_specs(numeric: list[str], categorical: list[str]):
    pre = make_preprocessor(numeric, categorical)

    return {
        "Ridge": (
            Pipeline([
                ("pre", pre),
                ("model", Ridge()),
            ]),
            {
                "model__alpha": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100,
                    1000,
                ]
            },
        ),
        "Random forest": (
            Pipeline([
                ("pre", pre),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]),
            {
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [5, 20, 50],
                "model__max_features": [0.5, 1.0],
            },
        ),
        "XGBoost": (
            Pipeline([
                ("pre", pre),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=300,
                        random_state=SEED,
                        n_jobs=-1,
                        verbosity=0,
                        colsample_bytree=0.8,
                    ),
                ),
            ]),
            {
                "model__max_depth": [2, 4],
                "model__learning_rate": [0.03, 0.07],
                "model__min_child_weight": [5, 20],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }


def fit_feature_set(
    rows: pd.DataFrame,
    feature_set: str,
    search_dir: Path,
):
    numeric = FEATURE_SETS[feature_set]["numeric"]
    categorical = FEATURE_SETS[feature_set]["categorical"]
    features = numeric + categorical

    dev = rows[
        rows["origin"].isin([TRAIN_ORIGIN, VALID_ORIGIN])
    ].reset_index(drop=True)

    test = rows[
        rows["origin"].eq(TEST_ORIGIN)
    ].reset_index(drop=True)

    train_idx = np.flatnonzero(
        dev["origin"].to_numpy() == TRAIN_ORIGIN
    )

    valid_idx = np.flatnonzero(
        dev["origin"].to_numpy() == VALID_ORIGIN
    )

    cv = [(train_idx, valid_idx)]

    selected = {}
    validation_predictions = {}
    holdout_predictions = {}
    rows_metrics = []

    specs = model_specs(numeric, categorical)

    search_dir.mkdir(parents=True, exist_ok=True)

    for model_name, (estimator, grid) in specs.items():
        print(
            f"TUNING {feature_set} / {model_name}",
            flush=True,
        )

        search = GridSearchCV(
            estimator,
            param_grid=grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            refit=False,
            n_jobs=1,
            error_score="raise",
        )

        search.fit(dev[features], dev["target"])

        search_table = pd.DataFrame(
            search.cv_results_
        )

        keep = [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_fit_time",
        ]

        search_table[keep].to_csv(
            search_dir
            / f"{slug(feature_set)}__{slug(model_name)}_search.csv",
            index=False,
        )

        selected[model_name] = {
            "best_params": search.best_params_,
            "validation_mae": float(-search.best_score_),
        }

        # Validation prediction uses only the 2021-origin rows for fitting.
        val_model = clone(estimator).set_params(
            **search.best_params_
        )

        val_model.fit(
            dev.loc[train_idx, features],
            dev.loc[train_idx, "target"],
        )

        validation_predictions[model_name] = np.clip(
            val_model.predict(
                dev.loc[valid_idx, features]
            ),
            0,
            None,
        )

        # Final model refits on both development origins,
        # but still never sees the 2024 holdout target.
        final_model = clone(estimator).set_params(
            **search.best_params_
        )

        final_model.fit(
            dev[features],
            dev["target"],
        )

        pred = np.clip(
            final_model.predict(test[features]),
            0,
            None,
        )

        holdout_predictions[model_name] = pred

        m = metrics(test["target"], pred)

        rows_metrics.append({
            "feature_set": feature_set,
            "model": model_name,
            "validation_MAE": float(-search.best_score_),
            **m,
        })

    # RF / XGB validation-selected blend.
    y_valid = dev.loc[valid_idx, "target"].to_numpy()

    rf_valid = validation_predictions["Random forest"]
    xgb_valid = validation_predictions["XGBoost"]

    candidates = []

    for w_rf in np.linspace(0, 1, 11):
        blend = (
            w_rf * rf_valid
            + (1 - w_rf) * xgb_valid
        )

        candidates.append({
            "weight_rf": float(w_rf),
            "weight_xgb": float(1 - w_rf),
            "validation_MAE": float(
                mean_absolute_error(y_valid, blend)
            ),
        })

    blend_table = pd.DataFrame(candidates).sort_values(
        "validation_MAE"
    )

    blend_table.to_csv(
        search_dir
        / f"{slug(feature_set)}__rf_xgb_blend_search.csv",
        index=False,
    )

    best_blend = blend_table.iloc[0]

    w_rf = float(best_blend["weight_rf"])
    w_xgb = float(best_blend["weight_xgb"])

    blend_pred = (
        w_rf * holdout_predictions["Random forest"]
        + w_xgb * holdout_predictions["XGBoost"]
    )

    holdout_predictions["RF-XGB blend"] = blend_pred

    blend_metrics = metrics(
        test["target"],
        blend_pred,
    )

    rows_metrics.append({
        "feature_set": feature_set,
        "model": "RF-XGB blend",
        "validation_MAE": float(
            best_blend["validation_MAE"]
        ),
        **blend_metrics,
    })

    selected["RF-XGB blend"] = {
        "weight_rf": w_rf,
        "weight_xgb": w_xgb,
        "validation_mae": float(
            best_blend["validation_MAE"]
        ),
    }

    return {
        "dev": dev,
        "test": test,
        "features": features,
        "selected": selected,
        "metrics": pd.DataFrame(rows_metrics),
        "predictions": holdout_predictions,
    }


def evaluate_sensitivity(
    rows: pd.DataFrame,
    primary_selected: dict,
):
    feature_set = "history_plus_census"

    numeric = FEATURE_SETS[feature_set]["numeric"]
    categorical = FEATURE_SETS[feature_set]["categorical"]
    features = numeric + categorical

    dev = rows[
        rows["origin"].isin([TRAIN_ORIGIN, VALID_ORIGIN])
    ].reset_index(drop=True)

    test = rows[
        rows["origin"].eq(TEST_ORIGIN)
    ].reset_index(drop=True)

    specs = model_specs(numeric, categorical)

    preds = {}

    out_rows = []

    baseline_prev = test["rate_lag1"].to_numpy()
    baseline_mean2 = test["rate_mean2"].to_numpy()

    for name, pred in [
        ("Previous year", baseline_prev),
        ("Mean previous 2 years", baseline_mean2),
    ]:
        out_rows.append({
            "model": name,
            **metrics(test["target"], pred),
        })

    for model_name in [
        "Ridge",
        "Random forest",
        "XGBoost",
    ]:
        estimator, _ = specs[model_name]

        best_params = primary_selected[
            model_name
        ]["best_params"]

        model = clone(estimator).set_params(
            **best_params
        )

        model.fit(
            dev[features],
            dev["target"],
        )

        pred = np.clip(
            model.predict(test[features]),
            0,
            None,
        )

        preds[model_name] = pred

        out_rows.append({
            "model": model_name,
            **metrics(test["target"], pred),
        })

    w_rf = primary_selected[
        "RF-XGB blend"
    ]["weight_rf"]

    w_xgb = primary_selected[
        "RF-XGB blend"
    ]["weight_xgb"]

    blend = (
        w_rf * preds["Random forest"]
        + w_xgb * preds["XGBoost"]
    )

    out_rows.append({
        "model": "RF-XGB blend",
        **metrics(test["target"], blend),
    })

    return pd.DataFrame(out_rows)


def log_existing_m6(run_index: list[dict]):
    exp1_metrics = pd.read_csv(
        ROOT
        / "results"
        / "m6"
        / "experiment1_expanded"
        / "holdout_model_metrics.csv"
    ).set_index("model")

    exp2_metrics = pd.read_csv(
        ROOT
        / "results"
        / "m6"
        / "experiment2_expanded"
        / "holdout_model_metrics.csv"
    ).set_index("model")

    with mlflow.start_run(
        run_name="M6 Experiment 1 verified summary"
    ) as run:
        mlflow.set_tags({
            "course": "BDA602",
            "experiment": "Experiment 1",
            "tracking_type": "verified existing output",
        })

        mlflow.log_param(
            "design",
            "updated-history one-month-ahead",
        )

        mlflow.log_metric(
            "random_forest_holdout_mae",
            float(
                exp1_metrics.loc[
                    "Random forest",
                    "MAE",
                ]
            ),
        )

        mlflow.log_metric(
            "random_forest_holdout_rmse",
            float(
                exp1_metrics.loc[
                    "Random forest",
                    "RMSE",
                ]
            ),
        )

        mlflow.log_metric(
            "previous_month_baseline_mae",
            float(
                exp1_metrics.loc[
                    "Previous month",
                    "MAE",
                ]
            ),
        )

        run_index.append({
            "experiment": "M6 Experiment 1",
            "feature_set": "",
            "model": "verified summary",
            "run_id": run.info.run_id,
        })

    with mlflow.start_run(
        run_name="M6 Experiment 2 verified summary"
    ) as run:
        mlflow.set_tags({
            "course": "BDA602",
            "experiment": "Experiment 2",
            "tracking_type": "verified existing output",
        })

        mlflow.log_param(
            "design",
            "fixed-origin end-2015",
        )

        mlflow.log_metric(
            "decision_tree_holdout_mae",
            float(
                exp2_metrics.loc[
                    "Decision tree",
                    "MAE",
                ]
            ),
        )

        mlflow.log_metric(
            "mean_2013_2015_baseline_mae",
            float(
                exp2_metrics.loc[
                    "Mean 2013–2015",
                    "MAE",
                ]
            ),
        )

        run_index.append({
            "experiment": "M6 Experiment 2",
            "feature_set": "",
            "model": "verified summary",
            "run_id": run.info.run_id,
        })


def log_m7_runs(
    all_results: dict,
    run_index: list[dict],
):
    for feature_set, result in all_results.items():
        metric_table = result["metrics"].set_index(
            "model"
        )

        for model_name, info in result[
            "selected"
        ].items():

            with mlflow.start_run(
                run_name=(
                    f"M7 Experiment 3 "
                    f"{feature_set} {model_name}"
                )
            ) as run:

                mlflow.set_tags({
                    "course": "BDA602",
                    "experiment": "Experiment 3",
                    "feature_set": feature_set,
                    "forecast_design": (
                        "one-year-ahead adoption growth"
                    ),
                    "holdout_year": str(TEST_YEAR),
                })

                mlflow.log_params({
                    "train_origin": TRAIN_ORIGIN,
                    "validation_origin": VALID_ORIGIN,
                    "test_origin": TEST_ORIGIN,
                    "min_dwellings": PRIMARY_MIN_DWELLINGS,
                    "small_pop_flag_required": 0,
                })

                if model_name == "RF-XGB blend":
                    mlflow.log_param(
                        "weight_rf",
                        info["weight_rf"],
                    )
                    mlflow.log_param(
                        "weight_xgb",
                        info["weight_xgb"],
                    )
                else:
                    for k, v in info[
                        "best_params"
                    ].items():
                        mlflow.log_param(
                            "selected_"
                            + k.replace("__", "_"),
                            str(v),
                        )

                row = metric_table.loc[model_name]

                mlflow.log_metric(
                    "validation_mae",
                    float(
                        info["validation_mae"]
                    ),
                )

                for metric_name in [
                    "MAE",
                    "RMSE",
                    "MedAE",
                    "R2",
                    "MAE_pct_points",
                    "RMSE_pct_points",
                ]:
                    mlflow.log_metric(
                        "holdout_"
                        + metric_name.lower(),
                        float(row[metric_name]),
                    )

                run_index.append({
                    "experiment": "M7 Experiment 3",
                    "feature_set": feature_set,
                    "model": model_name,
                    "run_id": run.info.run_id,
                })


def make_figures(
    metrics_table: pd.DataFrame,
    predictions: pd.DataFrame,
    ablation: pd.DataFrame,
):
    # 1. Holdout MAE comparison.
    plot = metrics_table.copy()

    labels = (
        plot["feature_set"]
        + " | "
        + plot["model"]
    )

    values = plot["MAE_pct_points"].to_numpy(dtype=float)
    y = np.arange(len(plot))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(values, y, s=55)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()

    for yi, value in zip(y, values):
        ax.annotate(
            f"{value:.3f}",
            (value, yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    xmin = min(1.0, float(values.min() - 0.02))
    xmax = max(1.21, float(values.max() + 0.02))
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(
        "2024 holdout MAE "
        "(percentage points of annual adoption growth)"
    )
    ax.set_title(
        "Experiment 3 — 2024 holdout MAE "
        "(zoomed scale; lower is better)"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUT / "fig_m7_holdout_mae_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    # 2. Best learned / ensemble prediction scatter.
    learned = metrics_table[
        metrics_table["feature_set"].ne("baseline")
    ].sort_values("MAE")

    best = learned.iloc[0]

    pred_col = (
        slug(best["feature_set"])
        + "__"
        + slug(best["model"])
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        predictions["actual"],
        predictions[pred_col],
        s=10,
        alpha=0.45,
    )

    lim = max(
        predictions["actual"].max(),
        predictions[pred_col].max(),
    )

    ax.plot(
        [0, lim],
        [0, lim],
        "--",
        linewidth=1,
    )

    ax.set_xlabel("Actual 2024 adoption-rate increment")
    ax.set_ylabel("Predicted 2024 adoption-rate increment")
    ax.set_title(
        f"Best learned/ensemble model: "
        f"{best['feature_set']} — {best['model']}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUT / "fig_m7_actual_vs_predicted.png",
        dpi=180,
    )
    plt.close(fig)

    # 3. Census-context ablation.
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.barh(
        ablation["model"],
        ablation["mae_delta_pct_points"],
    )

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlabel(
        "MAE change from adding Census context "
        "(percentage points; negative = improvement)"
    )

    ax.set_title(
        "Experiment 3 — Census-context ablation"
    )

    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUT / "fig_m7_census_ablation.png",
        dpi=180,
    )
    plt.close(fig)


def main():
    run_started = time.monotonic()

    OUT.mkdir(parents=True, exist_ok=True)

    # Preserve the committed human-readable results README across reruns.
    for existing in OUT.iterdir():
        if existing.name == "README.md":
            continue
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()

    search_dir = OUT / "search"
    search_dir.mkdir(exist_ok=True)

    print(
        "\n===== BUILD PRIMARY M7 SAMPLE =====",
        flush=True,
    )

    rows = build_rows(PRIMARY_MIN_DWELLINGS)

    counts = (
        rows.groupby("origin")[POSTCODE]
        .nunique()
        .to_dict()
    )

    print("postcodes by origin:", counts)

    if set(counts) != {
        TRAIN_ORIGIN,
        VALID_ORIGIN,
        TEST_ORIGIN,
    }:
        raise RuntimeError(
            "Unexpected origin coverage."
        )

    if min(counts.values()) < 2400:
        raise RuntimeError(
            "Primary sample unexpectedly small."
        )

    if not rows["small_pop_flag"].eq(0).all():
        raise RuntimeError(
            "Primary sample contains small_pop_flag rows."
        )

    if not rows[
        "dwellings_opd_2021"
    ].ge(PRIMARY_MIN_DWELLINGS).all():
        raise RuntimeError(
            "Primary sample contains small denominators."
        )

    all_results = {}

    for feature_set in FEATURE_SETS:
        print(
            f"\n===== {feature_set} =====",
            flush=True,
        )

        all_results[feature_set] = fit_feature_set(
            rows,
            feature_set,
            search_dir,
        )

    test = all_results[
        "history_only"
    ]["test"].copy()

    baseline_rows = []

    baseline_preds = {
        "Previous year": test[
            "rate_lag1"
        ].to_numpy(),
        "Mean previous 2 years": test[
            "rate_mean2"
        ].to_numpy(),
    }

    for name, pred in baseline_preds.items():
        baseline_rows.append({
            "feature_set": "baseline",
            "model": name,
            "validation_MAE": np.nan,
            **metrics(test["target"], pred),
        })

    metrics_table = pd.concat(
        [
            pd.DataFrame(baseline_rows),
            all_results[
                "history_only"
            ]["metrics"],
            all_results[
                "history_plus_census"
            ]["metrics"],
        ],
        ignore_index=True,
    )

    metrics_table.to_csv(
        OUT / "holdout_model_metrics.csv",
        index=False,
        float_format="%.8f",
    )

    # Holdout predictions.
    predictions = test[
        [
            POSTCODE,
            "origin",
            "target_year",
            "target",
            "dwellings_opd_2021",
        ]
    ].copy()

    predictions = predictions.rename(
        columns={"target": "actual"}
    )

    predictions[
        "baseline__previous_year"
    ] = baseline_preds["Previous year"]

    predictions[
        "baseline__mean_previous_2_years"
    ] = baseline_preds[
        "Mean previous 2 years"
    ]

    for feature_set, result in all_results.items():
        for model_name, pred in result[
            "predictions"
        ].items():

            col = (
                slug(feature_set)
                + "__"
                + slug(model_name)
            )

            predictions[col] = pred

    predictions.to_csv(
        OUT / "holdout_predictions_2024.csv",
        index=False,
        float_format="%.8f",
    )

    # Context ablation.
    ablation_rows = []

    for model_name in [
        "Ridge",
        "Random forest",
        "XGBoost",
        "RF-XGB blend",
    ]:
        h = metrics_table[
            (metrics_table["feature_set"] == "history_only")
            & (metrics_table["model"] == model_name)
        ].iloc[0]

        c = metrics_table[
            (metrics_table["feature_set"] == "history_plus_census")
            & (metrics_table["model"] == model_name)
        ].iloc[0]

        ablation_rows.append({
            "model": model_name,
            "history_only_MAE": float(h["MAE"]),
            "history_plus_census_MAE": float(c["MAE"]),
            "mae_delta": float(
                c["MAE"] - h["MAE"]
            ),
            "mae_delta_pct_points": float(
                100 * (c["MAE"] - h["MAE"])
            ),
        })

    ablation = pd.DataFrame(ablation_rows)

    ablation.to_csv(
        OUT / "census_context_ablation.csv",
        index=False,
        float_format="%.8f",
    )

    # ≥100-dwelling sensitivity analysis.
    print(
        "\n===== DENOMINATOR SENSITIVITY =====",
        flush=True,
    )

    rows100 = build_rows(
        SENSITIVITY_MIN_DWELLINGS
    )

    sensitivity = evaluate_sensitivity(
        rows100,
        all_results[
            "history_plus_census"
        ]["selected"],
    )

    sensitivity.insert(
        0,
        "min_dwellings",
        SENSITIVITY_MIN_DWELLINGS,
    )

    sensitivity.to_csv(
        OUT
        / "sensitivity_min_100_dwellings.csv",
        index=False,
        float_format="%.8f",
    )

    # Figures.
    make_figures(
        metrics_table,
        predictions,
        ablation,
    )

    # MLflow tracking.
    print(
        "\n===== MLFLOW TRACKING =====",
        flush=True,
    )

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.resolve()}"
    )

    mlflow.set_experiment(
        "BDA602 Solar Uninterrupted"
    )

    run_index = []

    log_existing_m6(run_index)
    log_m7_runs(all_results, run_index)

    pd.DataFrame(run_index).to_csv(
        OUT / "mlflow_run_index.csv",
        index=False,
    )

    # Metadata.
    best_learned = (
        metrics_table[
            metrics_table["feature_set"].ne(
                "baseline"
            )
        ]
        .sort_values("MAE")
        .iloc[0]
    )

    baseline_best = (
        metrics_table[
            metrics_table["feature_set"].eq(
                "baseline"
            )
        ]
        .sort_values("MAE")
        .iloc[0]
    )

    metadata = {
        "experiment": (
            "Experiment 3: one-year-ahead "
            "adoption-growth forecasting"
        ),
        "owner": "Shashwat Bajaj",
        "train_origin": TRAIN_ORIGIN,
        "validation_origin": VALID_ORIGIN,
        "test_origin": TEST_ORIGIN,
        "holdout_target_year": TEST_YEAR,
        "target": (
            "annual installations / "
            "2021 occupied private dwellings"
        ),
        "primary_sample": {
            "small_pop_flag": 0,
            "minimum_occupied_private_dwellings":
                PRIMARY_MIN_DWELLINGS,
            "postcodes_by_origin": {
                str(k): int(v)
                for k, v in counts.items()
            },
        },
        "sensitivity_minimum_dwellings":
            SENSITIVITY_MIN_DWELLINGS,
        "feature_sets": FEATURE_SETS,
        "models": [
            "Ridge",
            "Random forest",
            "XGBoost",
            "RF-XGB blend",
        ],
        "baselines": [
            "Previous year",
            "Mean previous 2 years",
        ],
        "selection_rule": (
            "Hyperparameters selected using "
            "origin 2021 -> target 2022 training "
            "and origin 2022 -> target 2023 "
            "chronological validation. "
            "Final models refit on both development "
            "origins and evaluated once on "
            "origin 2023 -> target 2024."
        ),
        "blend_rule": (
            "RF/XGBoost weight selected on "
            "2023 validation target only; "
            "candidate RF weights 0.0 to 1.0 "
            "in increments of 0.1."
        ),
        "census_boundary": (
            "The 2021 Census reference date predates all "
            "Experiment 3 target years. Most 2021 Census "
            "topics were publicly released on 28 June 2022. "
            "The 2021 -> 2022 development row is therefore "
            "retrospective with respect to publication "
            "availability, while the 2022 -> 2023 validation "
            "and 2023 -> 2024 holdout are out-of-time forecasts "
            "with Census data available by the forecasting "
            "origin. Census variables remain excluded from the "
            "2015-origin Experiment 2 forecast."
        ),
        "denominator_caution": (
            "Adoption-rate denominator is static "
            "2021 occupied private dwellings. "
            "Primary sample excludes small-population "
            "areas and denominators below 50; "
            "a >=100-dwelling sensitivity is reported."
        ),
        "best_learned_or_ensemble": {
            "feature_set": str(
                best_learned["feature_set"]
            ),
            "model": str(best_learned["model"]),
            "MAE": float(best_learned["MAE"]),
            "RMSE": float(best_learned["RMSE"]),
            "R2": float(best_learned["R2"]),
        },
        "best_naive_baseline": {
            "model": str(baseline_best["model"]),
            "MAE": float(baseline_best["MAE"]),
        },
        "elapsed_seconds": round(
            time.monotonic() - run_started,
            1,
        ),
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "xgboost": package_version("xgboost"),
            "mlflow": package_version("mlflow"),
            "matplotlib": package_version("matplotlib"),
        },
        "mlflow_tracking_uri": (
            "local gitignored SQLite database mlflow.db"
        ),
        "seed": SEED,
    }

    (
        OUT / "run_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(
        "\n===== HOLDOUT MODEL METRICS ====="
    )

    print(
        metrics_table[
            [
                "feature_set",
                "model",
                "validation_MAE",
                "MAE",
                "RMSE",
                "MedAE",
                "R2",
                "MAE_pct_points",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n===== CENSUS ABLATION ====="
    )

    print(
        ablation.round(6).to_string(
            index=False
        )
    )

    print(
        "\n===== >=100 DWELLING SENSITIVITY ====="
    )

    print(
        sensitivity[
            [
                "model",
                "MAE",
                "RMSE",
                "MedAE",
                "R2",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nBEST LEARNED/ENSEMBLE:",
        best_learned["feature_set"],
        "/",
        best_learned["model"],
        "MAE",
        round(
            float(best_learned["MAE"]),
            6,
        ),
    )

    print(
        "BEST BASELINE:",
        baseline_best["model"],
        "MAE",
        round(
            float(baseline_best["MAE"]),
            6,
        ),
    )


if __name__ == "__main__":
    main()
