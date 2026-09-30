#!/usr/bin/env python3
"""BDA 602 Solar Uninterrupted — Module 6 expanded tuning + ABS cluster profile.

Owner: Shashwat Bajaj (Modeling Lead)

This script extends the two Module 5 experiments without changing their forecasting
meaning:

Experiment 1: updated-history one-month-ahead postcode-month count backtest.
  - model weights fit on 2008–2015
  - each held-out month may use observed history through t-1
  - expanded Ridge grid and BayesSearchCV for Random Forest

Experiment 2: fixed-origin annual postcode count forecast from end-2015.
  - every 2016–2024 prediction uses information available at end-2015 only
  - expanded Ridge / Decision Tree grids and wider BayesSearchCV spaces for
    Random Forest and XGBoost

ABS profiling: joins the existing Experiment 2 residual clusters to the 2021 Census
features and the adoption-rate panel for descriptive post-hoc profiling only.
The 2021 Census variables are NOT treated as information available to a 2015
forecast. No causal claims are made.

Run from repository root after M5 Experiment 1 has been added:
    python src/m6_expanded_tuning.py --stage all

You can resume individual stages:
    python src/m6_expanded_tuning.py --stage exp1
    python src/m6_expanded_tuning.py --stage exp2
    python src/m6_expanded_tuning.py --stage abs

Requires scikit-optimize (already listed in repository requirements.txt).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
except Exception as exc:  # fail loudly; M6 specifically requires the Bayesian search
    raise SystemExit(
        "scikit-optimize is required for the M6 run. From the repository root run: "
        "python3 -m pip install -r requirements.txt"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
POSTCODE = "Small Unit Installation Postcode"
OUTROOT = ROOT / "results" / "m6"


def _load_m5_module():
    path = ROOT / "src" / "run_m5_count_analysis.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Merge/copy Experiment 1 M5 files before running M6."
        )
    spec = importlib.util.spec_from_file_location("m5_exp1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    y = np.asarray(y, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "actual_total": float(y.sum()),
        "predicted_total": float(p.sum()),
    }


def _jsonable_params(params):
    out = {}
    for k, v in params.items():
        if hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out


def _save_search(search, path: Path, name: str, search_type: str):
    tab = pd.DataFrame(search.cv_results_)
    cols = [
        c for c in [
            "params", "mean_test_score", "std_test_score", "rank_test_score",
            "split0_test_score", "split1_test_score", "mean_fit_time"
        ] if c in tab.columns
    ]
    out = tab[cols].copy()
    out.insert(0, "model", name)
    out["search"] = search_type
    out.to_csv(path, index=False)


def run_exp1(iterations: int = 16):
    print("\n===== M6 EXPERIMENT 1: EXPANDED TUNING =====", flush=True)
    m5 = _load_m5_module()
    out = OUTROOT / "experiment1_expanded"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    df, excluded = m5.load_features(ROOT / "data" / "processed" / "cer_solar_panel_long.csv")
    train = df.loc[df.month.lt("2016-01-01")].copy()
    test = df.loc[df.month.ge("2016-01-01")].copy()
    X = train[m5.FEATURES].to_numpy(dtype="float32")
    y = train.installations.to_numpy(dtype="float32")
    XT = test[m5.FEATURES].to_numpy(dtype="float32")
    yt = test.installations.to_numpy(dtype="float32")
    dates = train.month.to_numpy()
    folds = [
        (np.flatnonzero(dates < np.datetime64("2014-01-01")),
         np.flatnonzero((dates >= np.datetime64("2014-01-01")) & (dates < np.datetime64("2015-01-01")))),
        (np.flatnonzero(dates < np.datetime64("2015-01-01")),
         np.flatnonzero((dates >= np.datetime64("2015-01-01")) & (dates < np.datetime64("2016-01-01")))),
    ]
    assert all(dates[a].max() < dates[b].min() for a, b in folds)
    assert len(train) == 269664 and len(test) == 303480

    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    ridge_grid = {"model__alpha": [0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 500, 1000]}
    ridge_search = GridSearchCV(
        ridge, ridge_grid, scoring="neg_mean_absolute_error", cv=folds,
        refit=False, n_jobs=1, error_score="raise"
    )
    t = time.monotonic(); ridge_search.fit(X, y)
    _save_search(ridge_search, out / "ridge_search.csv", "Ridge", "GridSearchCV")
    print("Ridge best", ridge_search.best_params_, "CV MAE", -ridge_search.best_score_,
          "seconds", round(time.monotonic() - t, 1), flush=True)

    # More trees than M5 (150 vs 25) and a substantially wider search space.
    rf = RandomForestRegressor(
        random_state=SEED, n_jobs=-1, n_estimators=150, bootstrap=True
    )
    rf_space = {
        "max_depth": Integer(6, 24),
        "min_samples_leaf": Integer(2, 40),
        "max_features": Real(0.30, 1.00),
        "max_samples": Real(0.50, 0.90),
    }
    rf_search = BayesSearchCV(
        rf, rf_space, n_iter=iterations, cv=folds,
        scoring="neg_mean_absolute_error", refit=False,
        random_state=SEED, n_jobs=1, error_score="raise"
    )
    t = time.monotonic(); rf_search.fit(X, y)
    _save_search(rf_search, out / "random_forest_search.csv", "Random forest", "BayesSearchCV")
    print("Random forest best", rf_search.best_params_, "CV MAE", -rf_search.best_score_,
          "seconds", round(time.monotonic() - t, 1), flush=True)

    ridge_est = clone(ridge).set_params(**ridge_search.best_params_).fit(X, y)
    rf_est = clone(rf).set_params(**rf_search.best_params_).fit(X, y)
    preds = {
        "Previous month": test.lag_1.to_numpy(dtype=float),
        "Same month last year": test.lag_12.to_numpy(dtype=float),
        "Ridge": np.clip(ridge_est.predict(XT), 0, None),
        "Random forest": np.clip(rf_est.predict(XT), 0, None),
    }
    overall = pd.DataFrame([
        dict(model=name, observations=len(yt), **_metrics(yt, pred))
        for name, pred in preds.items()
    ])
    overall.to_csv(out / "holdout_model_metrics.csv", index=False, float_format="%.6f")

    byyear = []
    years = test.month.dt.year.to_numpy()
    for year in range(2016, 2025):
        mask = years == year
        for name, pred in preds.items():
            byyear.append(dict(year=year, model=name, **_metrics(yt[mask], pred[mask])))
    pd.DataFrame(byyear).to_csv(out / "holdout_metrics_by_year.csv", index=False, float_format="%.6f")

    prior_path = ROOT / "results" / "experiment1" / "holdout_model_metrics.csv"
    comparison = None
    if prior_path.exists():
        old = pd.read_csv(prior_path).set_index("model")
        new = overall.set_index("model")
        rows = []
        for model in ["Ridge", "Random forest"]:
            if model in old.index and model in new.index:
                rows.append({
                    "model": model,
                    "m5_mae": float(old.loc[model, "MAE"]),
                    "m6_mae": float(new.loc[model, "MAE"]),
                    "mae_change": float(new.loc[model, "MAE"] - old.loc[model, "MAE"]),
                    "m5_rmse": float(old.loc[model, "RMSE"]),
                    "m6_rmse": float(new.loc[model, "RMSE"]),
                })
        comparison = pd.DataFrame(rows)
        comparison.to_csv(out / "m5_vs_m6_metrics.csv", index=False, float_format="%.6f")

    meta = {
        "experiment": "Experiment 1 expanded tuning",
        "forecast_design": "updated-history one-month-ahead backtest; weights frozen after 2015",
        "train": "2008-01 through 2015-12",
        "holdout": "2016-01 through 2024-12",
        "rows_train": int(len(train)),
        "rows_holdout": int(len(test)),
        "postcodes": int(test.postcode.nunique()),
        "excluded_0000_installations": int(excluded),
        "validation": "two expanding full-year folds: <=2013 -> 2014; <=2014 -> 2015",
        "ridge_search": "GridSearchCV expanded alpha grid (11 candidates)",
        "random_forest_search": f"BayesSearchCV n_iter={iterations}; 150 trees per fit",
        "ridge_best_params": _jsonable_params(ridge_search.best_params_),
        "ridge_cv_mae": float(-ridge_search.best_score_),
        "rf_best_params": _jsonable_params(rf_search.best_params_),
        "rf_cv_mae": float(-rf_search.best_score_),
        "heldout_predictions_clipped_nonnegative": True,
        "seed": SEED,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(overall.round(4).to_string(index=False), flush=True)
    return {"metrics": overall, "meta": meta, "comparison": comparison}


def _load_exp2_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import src.fixed_origin_forecast as exp2
    return exp2


def run_exp2(iterations: int = 20):
    print("\n===== M6 EXPERIMENT 2: EXPANDED TUNING =====", flush=True)
    exp2 = _load_exp2_module()
    out = OUTROOT / "experiment2_expanded"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    a = exp2.annual_panel(ROOT / "data" / "processed" / "cer_solar_panel_long.csv")
    dev = exp2.build_rows(a, exp2.TRAIN_ORIGINS, exp2.ORIGIN_TEST)
    test = exp2.build_rows(a, [exp2.ORIGIN_TEST], max(exp2.TEST_YEARS))
    ylog = np.log1p(dev.target.to_numpy(dtype=float))
    yt = test.target.to_numpy(dtype=float)
    origins = dev.origin.to_numpy()
    folds = [
        (np.flatnonzero(origins <= 2012), np.flatnonzero(origins == 2013)),
        (np.flatnonzero(origins <= 2013), np.flatnonzero(origins == 2014)),
    ]

    pre = ColumnTransformer([
        ("num", StandardScaler(), exp2.NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), exp2.CAT_FEATURES),
    ])
    specs = {
        "Ridge": (
            Pipeline([("pre", pre), ("m", Ridge())]),
            "grid",
            {"m__alpha": [0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000]},
        ),
        "Decision tree": (
            Pipeline([("pre", pre), ("m", DecisionTreeRegressor(random_state=SEED))]),
            "grid",
            {"m__max_depth": [3, 4, 5, 6, 8, 10, 12, 16],
             "m__min_samples_leaf": [10, 20, 40, 60, 80, 100, 150]},
        ),
        "Random forest": (
            Pipeline([("pre", pre), ("m", RandomForestRegressor(
                random_state=SEED, n_jobs=-1, n_estimators=200
            ))]),
            "bayes",
            {"m__max_depth": Integer(3, 24),
             "m__min_samples_leaf": Integer(3, 150),
             "m__max_features": Real(0.20, 1.00)},
        ),
        "XGBoost": (
            Pipeline([("pre", pre), ("m", XGBRegressor(
                random_state=SEED, n_jobs=-1, n_estimators=300, verbosity=0
            ))]),
            "bayes",
            {"m__max_depth": Integer(2, 10),
             "m__learning_rate": Real(0.01, 0.30, prior="log-uniform"),
             "m__subsample": Real(0.50, 1.00),
             "m__min_child_weight": Integer(1, 30),
             "m__colsample_bytree": Real(0.50, 1.00)},
        ),
    }

    tuned = {}
    best_params = {}
    cv_scores = {}
    search_desc = {}
    for name, (pipe, kind, space) in specs.items():
        print("TUNING", name, flush=True)
        t = time.monotonic()
        if kind == "grid":
            search = GridSearchCV(
                pipe, space, cv=folds, scoring="neg_mean_absolute_error",
                refit=False, n_jobs=1, error_score="raise"
            )
            stype = "GridSearchCV"
        else:
            search = BayesSearchCV(
                pipe, space, n_iter=iterations, cv=folds,
                scoring="neg_mean_absolute_error", refit=False,
                random_state=SEED, n_jobs=1, error_score="raise"
            )
            stype = "BayesSearchCV"
        search.fit(dev, ylog)
        _save_search(search, out / f"{name.replace(' ', '_').lower()}_search.csv", name, stype)
        best_params[name] = _jsonable_params(search.best_params_)
        cv_scores[name] = float(-search.best_score_)
        search_desc[name] = stype if kind == "grid" else f"{stype} n_iter={iterations}"
        tuned[name] = clone(pipe).set_params(**search.best_params_).fit(dev, ylog)
        print(name, "best", search.best_params_, "CV log-MAE", -search.best_score_,
              "seconds", round(time.monotonic() - t, 1), flush=True)

    preds = {
        "Persistence (2015 count)": test.lag1.to_numpy(dtype=float),
        "Mean 2013–2015": test.mean3.to_numpy(dtype=float),
    }
    for name, model in tuned.items():
        preds[name] = np.clip(np.expm1(model.predict(test)), 0, None)

    overall = pd.DataFrame([
        dict(model=name, **_metrics(yt, pred)) for name, pred in preds.items()
    ])
    overall.to_csv(out / "holdout_model_metrics.csv", index=False, float_format="%.6f")
    byyear = []
    for year in exp2.TEST_YEARS:
        mask = test.year.to_numpy() == year
        for name, pred in preds.items():
            byyear.append(dict(
                year=year, horizon=year-exp2.ORIGIN_TEST, model=name,
                **_metrics(yt[mask], pred[mask])
            ))
    pd.DataFrame(byyear).to_csv(out / "holdout_metrics_by_year.csv", index=False, float_format="%.6f")

    prior_path = ROOT / "results" / "experiment2" / "holdout_model_metrics.csv"
    comparison = None
    if prior_path.exists():
        old = pd.read_csv(prior_path).set_index("model")
        new = overall.set_index("model")
        rows = []
        for model in ["Ridge", "Decision tree", "Random forest", "XGBoost"]:
            if model in old.index and model in new.index:
                rows.append({
                    "model": model,
                    "m5_mae": float(old.loc[model, "MAE"]),
                    "m6_mae": float(new.loc[model, "MAE"]),
                    "mae_change": float(new.loc[model, "MAE"] - old.loc[model, "MAE"]),
                    "m5_rmse": float(old.loc[model, "RMSE"]),
                    "m6_rmse": float(new.loc[model, "RMSE"]),
                })
        comparison = pd.DataFrame(rows)
        comparison.to_csv(out / "m5_vs_m6_metrics.csv", index=False, float_format="%.6f")

    best_learned = overall[overall.model.isin(tuned)].sort_values("MAE").iloc[0].model
    meta = {
        "experiment": "Experiment 2 expanded tuning",
        "forecast_design": "fixed-origin annual forecast from end-2015",
        "dev_rows": int(len(dev)),
        "test_rows": int(len(test)),
        "postcodes": int(a.shape[0]),
        "test_years": list(exp2.TEST_YEARS),
        "validation": "two expanding-origin folds: origin<=2012 -> 2013; origin<=2013 -> 2014",
        "searches": search_desc,
        "best_params": best_params,
        "cv_log_mae": cv_scores,
        "best_learned_model_by_holdout_mae": best_learned,
        "naive_mean_holdout_mae": float(overall.set_index("model").loc["Mean 2013–2015", "MAE"]),
        "seed": SEED,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(overall.round(4).to_string(index=False), flush=True)
    return {"metrics": overall, "meta": meta, "comparison": comparison}


def run_abs_profile():
    print("\n===== M6 ABS / RESIDUAL-CLUSTER DESCRIPTIVE PROFILE =====", flush=True)
    out = OUTROOT / "abs_cluster_profile"
    out.mkdir(parents=True, exist_ok=True)

    clusters_path = ROOT / "results" / "experiment2" / "postcode_residual_clusters.csv"
    features_path = ROOT / "data" / "processed" / "abs_poa_features.csv"
    joined_path = ROOT / "data" / "processed" / "cer_abs_panel_long.csv.gz"
    for p in [clusters_path, features_path, joined_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required M6 input missing: {p}")

    clusters = pd.read_csv(clusters_path, dtype={"postcode": str})
    features = pd.read_csv(features_path, dtype={POSTCODE: str})
    features = features.rename(columns={POSTCODE: "postcode"})
    prof = clusters[["postcode", "cluster", "mean_rel_residual", "mean_abs_rel_residual"]].merge(
        features, on="postcode", how="left", validate="one_to_one"
    )

    joined = pd.read_csv(joined_path, dtype={POSTCODE: str}, parse_dates=["month"])
    snap = joined.loc[joined.month.eq(pd.Timestamp("2024-12-01")),
                      [POSTCODE, "adoption_rate", "dwellings_opd_2021"]].rename(columns={POSTCODE: "postcode"})
    prof = prof.merge(snap, on="postcode", how="left", validate="one_to_one")

    selected = [
        "Median_tot_hhd_inc_weekly", "Median_age_persons", "Average_household_size",
        "pop_density_per_sqkm", "detached_share", "apartment_share",
        "owner_occupied_share", "rented_share", "Median_mortgage_repay_monthly",
        "Median_rent_weekly", "adoption_rate"
    ]
    selected = [c for c in selected if c in prof.columns]
    agg = prof.groupby("cluster")[selected].agg(["count", "mean", "median"])
    agg.columns = [f"{c}__{stat}" for c, stat in agg.columns]
    agg.reset_index().to_csv(out / "cluster_feature_summary.csv", index=False, float_format="%.6f")

    effects = []
    clusters_present = sorted(prof.cluster.dropna().unique())
    if len(clusters_present) == 2:
        c0, c1 = clusters_present
        for col in selected:
            a = prof.loc[prof.cluster.eq(c0), col].dropna().astype(float)
            b = prof.loc[prof.cluster.eq(c1), col].dropna().astype(float)
            pooled = prof[col].dropna().astype(float).std(ddof=0)
            effects.append({
                "feature": col,
                "cluster0_n": int(len(a)), "cluster1_n": int(len(b)),
                "cluster0_mean": float(a.mean()), "cluster1_mean": float(b.mean()),
                "cluster0_median": float(a.median()), "cluster1_median": float(b.median()),
                "standardized_mean_difference_c1_minus_c0": float((b.mean()-a.mean())/pooled) if pooled > 0 else np.nan,
            })
    effects = pd.DataFrame(effects).sort_values(
        "standardized_mean_difference_c1_minus_c0", key=lambda s: s.abs(), ascending=False
    )
    effects.to_csv(out / "cluster_feature_effects.csv", index=False, float_format="%.6f")
    prof.to_csv(out / "postcode_cluster_abs_profile.csv", index=False, float_format="%.6f")

    if not effects.empty:
        top = effects.head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(top.feature, top.standardized_mean_difference_c1_minus_c0)
        ax.axvline(0, linewidth=1, linestyle="--")
        ax.set(
            title="Experiment 2 residual clusters — descriptive ABS feature differences",
            xlabel="Standardized mean difference: Cluster 1 − Cluster 0",
            ylabel=""
        )
        fig.tight_layout()
        fig.savefig(out / "fig_abs_cluster_standardized_differences.png", dpi=175)
        plt.close(fig)

    meta = {
        "analysis": "post-hoc descriptive profiling of Experiment 2 residual clusters",
        "cluster_input": "results/experiment2/postcode_residual_clusters.csv",
        "census_features": "ABS 2021 Census static snapshot",
        "adoption_snapshot": "2024-12 cumulative installations / 2021 occupied private dwellings",
        "interpretation_rule": "descriptive associations only; not causal and not information available to a 2015 forecast",
        "clustered_postcodes": int(len(clusters)),
        "matched_feature_rows": int(prof[selected].notna().any(axis=1).sum()),
        "selected_features": selected,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not effects.empty:
        print(effects.head(8).round(4).to_string(index=False), flush=True)
    return {"effects": effects, "meta": meta}


def write_master_summary(exp1=None, exp2=None, abs_profile=None):
    OUTROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_unix": int(time.time()),
        "python": platform.python_version(),
        "seed": SEED,
        "exp1": exp1["meta"] if exp1 else None,
        "exp2": exp2["meta"] if exp2 else None,
        "abs_profile": abs_profile["meta"] if abs_profile else None,
    }
    (OUTROOT / "m6_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["exp1", "exp2", "abs", "all"], default="all")
    ap.add_argument("--exp1-iterations", type=int, default=16)
    ap.add_argument("--exp2-iterations", type=int, default=20)
    args = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    e1 = e2 = ab = None
    if args.stage in ("exp1", "all"):
        e1 = run_exp1(args.exp1_iterations)
    if args.stage in ("exp2", "all"):
        e2 = run_exp2(args.exp2_iterations)
    if args.stage in ("abs", "all"):
        ab = run_abs_profile()
    write_master_summary(e1, e2, ab)
    print("\nM6 RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
