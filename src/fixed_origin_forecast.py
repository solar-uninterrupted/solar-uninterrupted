#!/usr/bin/env python3
"""
fixed_origin_forecast.py — Solar Uninterrupted, Experiment 2

Fixed-origin annual forecast: every prediction for 2016–2024 uses ONLY
information available at the end of 2015. This is the design in the M4 paper
outline (train 2001–2015 → predict 2016–2026) and the complement to Experiment 1
(run_m5_count_analysis.py), which is a one-month-ahead backtest that refreshes
lags with post-2015 actuals.

Design
------
Unit: postcode-year. Target: annual installation count in year Y (modelled on
log1p, reported on the raw scale). Horizon h = Y − origin.

Training rows are built from historical origins o ∈ {2006, …, 2014} with all
targets o+1 … 2015 (h = 1 … 9), so the model learns how counts evolve h years
after a snapshot using only pre-2016 data. Features are computed strictly from
years ≤ o. At test time the origin is 2015 and h runs 1 … 9 (2016–2024).
2025–2026 are excluded (registration lag).

Validation (tuning): two expanding-origin folds inside the development data —
train on rows with origin ≤ 2012, validate on origin = 2013 rows; train on
origin ≤ 2013, validate on origin = 2014. No random splits. Models are refit on
all development rows and scored ONCE on the 2015-origin test rows.

Models: ridge (baseline), decision tree, random forest, XGBoost. Baselines:
persistence (2015 count carried forward) and 2013–2015 mean.

Unsupervised: for each postcode, the 9-year profile of RELATIVE residuals
(actual − predicted) / (1 + mean annual count 2013–2015) from the best model,
standardized across postcodes, PCA → K-Means, k chosen by silhouette.
Relative scaling is deliberate: it stops high-volume postcodes dominating the
clusters (see run_metadata.json for the rationale).

Usage: python src/fixed_origin_forecast.py --input data/processed/cer_solar_panel_long.csv --out results/experiment2
"""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import clone
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real

warnings.filterwarnings("ignore")
SEED = 42
POSTCODE = "Small Unit Installation Postcode"
ORIGIN_TEST = 2015
TEST_YEARS = list(range(2016, 2025))
TRAIN_ORIGINS = list(range(2006, 2015))

STATE_BY_PREFIX = {"0": "NT/ACT", "1": "NSW", "2": "NSW/ACT", "3": "VIC", "4": "QLD",
                   "5": "SA", "6": "WA", "7": "TAS", "8": "VIC", "9": "QLD"}

NUM_FEATURES = ["lag1", "lag2", "lag3", "mean3", "cum", "years_active", "growth", "share", "log_cum", "horizon"]
CAT_FEATURES = ["state"]


def annual_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={POSTCODE: str}, parse_dates=["month"])
    df = df[df[POSTCODE] != "0000"]
    df["year"] = df.month.dt.year
    a = df.groupby([POSTCODE, "year"]).installations.sum().unstack("year").fillna(0)
    a = a.loc[:, [y for y in a.columns if y <= 2024]]
    return a  # postcodes × years, integer counts


def snapshot_features(a: pd.DataFrame, origin: int) -> pd.DataFrame:
    """Features for every postcode using only years <= origin."""
    yrs = [y for y in a.columns if y <= origin]
    hist = a[yrs]
    lag1, lag2, lag3 = hist[origin], hist[origin - 1], hist[origin - 2]
    mean3 = (lag1 + lag2 + lag3) / 3
    cum = hist.sum(axis=1)
    first = hist.gt(0).idxmax(axis=1).where(hist.gt(0).any(axis=1), np.nan)
    years_active = (origin - first).fillna(0)
    growth = (lag1 + 1) / (lag2 + 1)
    share = lag1 / max(lag1.sum(), 1)
    f = pd.DataFrame({"lag1": lag1, "lag2": lag2, "lag3": lag3, "mean3": mean3, "cum": cum,
                      "years_active": years_active, "growth": growth, "share": share,
                      "log_cum": np.log1p(cum)})
    f["state"] = f.index.str[0].map(STATE_BY_PREFIX).fillna("OTHER")
    f["origin"] = origin
    return f


def build_rows(a: pd.DataFrame, origins, max_year) -> pd.DataFrame:
    rows = []
    for o in origins:
        f = snapshot_features(a, o)
        for y in range(o + 1, max_year + 1):
            r = f.copy(); r["year"] = y; r["horizon"] = y - o; r["target"] = a[y].values
            rows.append(r)
    out = pd.concat(rows).reset_index().rename(columns={POSTCODE: "postcode"})
    return out


def metrics(y, p):
    p = np.clip(p, 0, None)
    return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)), actual_total=int(y.sum()), predicted_total=float(p.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/processed/cer_solar_panel_long.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/experiment2"))
    ap.add_argument("--stage", choices=["tune", "fit", "all"], default="all")
    ap.add_argument("--model", default=None, help="model name to tune when --stage tune")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    a = annual_panel(args.input)
    dev = build_rows(a, TRAIN_ORIGINS, ORIGIN_TEST)          # all targets ≤ 2015
    test = build_rows(a, [ORIGIN_TEST], max(TEST_YEARS))     # targets 2016–2024
    assert dev.year.max() == 2015 and test.year.min() == 2016
    print(f"dev rows={len(dev):,} test rows={len(test):,} postcodes={a.shape[0]:,}")

    pre = ColumnTransformer([("num", StandardScaler(), NUM_FEATURES),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)])
    ylog = np.log1p(dev.target.values)
    # expanding-origin folds
    o = dev.origin.values
    folds = [(np.flatnonzero(o <= 2012), np.flatnonzero(o == 2013)),
             (np.flatnonzero(o <= 2013), np.flatnonzero(o == 2014))]

    specs = {
        "Ridge": (Pipeline([("pre", pre), ("m", Ridge())]),
                  "grid", {"m__alpha": [0.1, 1, 10, 100]}),
        "Decision tree": (Pipeline([("pre", pre), ("m", DecisionTreeRegressor(random_state=SEED))]),
                          "grid", {"m__max_depth": [4, 6, 8, 12], "m__min_samples_leaf": [20, 50, 100]}),
        "Random forest": (Pipeline([("pre", pre), ("m", RandomForestRegressor(random_state=SEED, n_jobs=-1, n_estimators=80))]),
                          "bayes", {"m__max_depth": Integer(4, 16), "m__min_samples_leaf": Integer(5, 100),
                                    "m__max_features": Real(0.3, 1.0)}),
        "XGBoost": (Pipeline([("pre", pre), ("m", XGBRegressor(random_state=SEED, n_jobs=-1, n_estimators=200, verbosity=0))]),
                    "bayes", {"m__max_depth": Integer(2, 8), "m__learning_rate": Real(0.02, 0.3, prior="log-uniform"),
                              "m__subsample": Real(0.5, 1.0), "m__min_child_weight": Integer(1, 20)}),
    }
    tune_dir = args.out / "tuning"; tune_dir.mkdir(exist_ok=True)
    if args.stage in ("tune", "all"):
        names = [args.model] if (args.stage == "tune" and args.model) else list(specs)
        for name in names:
            pipe, kind, space = specs[name]; t = time.monotonic()
            if kind == "grid":
                s = GridSearchCV(pipe, space, cv=folds, scoring="neg_mean_absolute_error", refit=False, n_jobs=1)
            else:
                s = BayesSearchCV(pipe, space, n_iter=12, cv=folds, scoring="neg_mean_absolute_error",
                                  refit=False, random_state=SEED, n_jobs=1)
            s.fit(dev, ylog)
            tab = pd.DataFrame(s.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]]
            tab.insert(0, "model", name); tab["search"] = "GridSearchCV" if kind == "grid" else "BayesSearchCV"
            tab.to_csv(tune_dir / f"{name.replace(' ', '_').lower()}_search.csv", index=False)
            bp = {k: (v.item() if hasattr(v, "item") else v) for k, v in s.best_params_.items()}
            (tune_dir / f"{name.replace(' ', '_').lower()}_best.json").write_text(
                json.dumps({"best_params": bp, "cv_log_mae": float(-s.best_score_)}, indent=2))
            print(f"{name}: best CV log-MAE={-s.best_score_:.4f} {bp} [{time.monotonic()-t:.0f}s]", flush=True)
        if args.stage == "tune":
            return
    tuned, search_tables, best_params, cv_scores = {}, [], {}, {}
    for name, (pipe, kind, space) in specs.items():
        slug = name.replace(" ", "_").lower()
        info = json.loads((tune_dir / f"{slug}_best.json").read_text())
        best_params[name] = info["best_params"]; cv_scores[name] = info["cv_log_mae"]
        search_tables.append(pd.read_csv(tune_dir / f"{slug}_search.csv"))
        t = time.monotonic(); tuned[name] = clone(pipe).set_params(**info["best_params"]).fit(dev, ylog)
        print(f"fit {name} [{time.monotonic()-t:.0f}s]", flush=True)
    pd.concat(search_tables).to_csv(args.out / "tuning_results.csv", index=False)

    yt = test.target.values.astype(float)
    preds = {"Persistence (2015 count)": test.lag1.values.astype(float),
             "Mean 2013–2015": test.mean3.values.astype(float)}
    for name, est in tuned.items():
        preds[name] = np.clip(np.expm1(est.predict(test)), 0, None)

    overall = pd.DataFrame([dict(model=n, **metrics(yt, p)) for n, p in preds.items()])
    overall.to_csv(args.out / "holdout_model_metrics.csv", index=False, float_format="%.4f")
    byyear = []
    for y in TEST_YEARS:
        m = test.year.values == y
        for n, p in preds.items():
            byyear.append(dict(year=y, horizon=y - ORIGIN_TEST, model=n, **metrics(yt[m], p[m])))
    byyear = pd.DataFrame(byyear); byyear.to_csv(args.out / "holdout_metrics_by_year.csv", index=False, float_format="%.4f")

    fc = test[["postcode", "year", "horizon", "target", "state", "mean3"]].copy()
    for n, p in preds.items(): fc[n] = p
    fc.to_csv(args.out / "holdout_predictions.csv.gz", index=False, compression="gzip", float_format="%.4f")

    best = overall[overall.model.isin(tuned)].sort_values("MAE").iloc[0].model
    print("best model by MAE:", best)

    # ---------- figures ----------
    nat = fc.groupby("year")[["target"] + list(preds)].sum()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(nat.index, nat["target"], "k-o", lw=2.2, label="Actual")
    for n in preds: ax.plot(nat.index, nat[n], "--o", lw=1.2, ms=4, label=n)
    ax.set_ylim(0, nat["target"].max() * 1.25)
    ax.text(0.99, 0.97, "Ridge (log-scale extrapolation) exceeds axis; see table", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray")
    ax.set(title="Experiment 2 — fixed-origin forecast from end-2015: national annual installations",
           xlabel="Year", ylabel="Installations (sum over postcodes)"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.out / "fig_national_actual_vs_forecast.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for n, seg in byyear.groupby("model", sort=False):
        ax.plot(seg.horizon, seg.MAE, "-o", label=n, lw=2 if n == best else 1.2)
    ax.set_ylim(0, 120)
    ax.text(0.99, 0.97, "Ridge exceeds axis beyond h=6; see table", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")
    ax.set(title="Experiment 2 — MAE by forecast horizon (years beyond 2015)", xlabel="Horizon (years)",
           ylabel="MAE (installations / postcode-year)"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.out / "fig_mae_by_horizon.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, y in zip(axes, [2016, 2024]):
        m = fc.year == y
        ax.scatter(fc.loc[m, best], fc.loc[m, "target"], s=6, alpha=.35)
        lim = max(fc.loc[m, "target"].max(), fc.loc[m, best].max())
        ax.plot([0, lim], [0, lim], "r--", lw=1); ax.set(xscale="symlog", yscale="symlog",
            title=f"{best}: actual vs forecast, {y} (h={y-2015})", xlabel="Forecast", ylabel="Actual")
    fig.tight_layout(); fig.savefig(args.out / "fig_actual_vs_forecast_scatter.png", dpi=170); plt.close(fig)

    # ---------- residual clustering (relative residuals) ----------
    fc["residual"] = fc["target"] - fc[best]
    fc["rel_residual"] = fc["residual"] / np.maximum(fc["mean3"], 10.0)
    cum2015 = a[[y for y in a.columns if y <= 2015]].sum(axis=1)
    active = cum2015[cum2015 >= 100].index            # postcodes with real activity before the origin
    prof = fc[fc.postcode.isin(active)].pivot(index="postcode", columns="year", values="rel_residual")
    prof = prof.clip(lower=prof.quantile(0.005).min(), upper=prof.quantile(0.995).max())  # tame extreme tails
    assert prof.shape[1] == len(TEST_YEARS) and not prof.isna().any().any()
    print(f"clustering {len(prof):,} active postcodes (>=100 installations through 2015)")
    Z = StandardScaler().fit_transform(prof.values)
    pca = PCA(n_components=3, random_state=SEED).fit(Z); E = pca.transform(Z)
    diag, kms = [], {}
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(E)
        diag.append(dict(k=k, silhouette=float(silhouette_score(E, km.labels_, sample_size=1500, random_state=SEED)),
                         inertia=float(km.inertia_))); kms[k] = km
    diag = pd.DataFrame(diag); diag.to_csv(args.out / "residual_cluster_k_diagnostics.csv", index=False)
    k = int(diag.loc[diag.silhouette.idxmax(), "k"]); labels = kms[k].labels_
    cl = prof.copy(); cl["cluster"] = labels
    cl["mean_rel_residual"] = prof.mean(axis=1); cl["mean_abs_rel_residual"] = prof.abs().mean(axis=1)
    cl["state"] = cl.index.str[0].map(STATE_BY_PREFIX).fillna("OTHER")
    cl["mean3_2013_2015"] = fc.drop_duplicates("postcode").set_index("postcode").loc[cl.index, "mean3"]
    cl.reset_index().to_csv(args.out / "postcode_residual_clusters.csv", index=False, float_format="%.5f")
    summ = cl.groupby("cluster").agg(postcodes=("state", "size"), mean_rel_residual=("mean_rel_residual", "mean"),
                                     mean_abs_rel_residual=("mean_abs_rel_residual", "mean"),
                                     median_volume_2013_2015=("mean3_2013_2015", "median"))
    state_mix = pd.crosstab(cl.cluster, cl.state, normalize="index").round(3)
    summ.to_csv(args.out / "residual_cluster_summary.csv"); state_mix.to_csv(args.out / "residual_cluster_state_mix.csv")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for c in sorted(set(labels)):
        v = prof[labels == c].mean(axis=0)
        ax.plot(v.index, v.values, "-o", label=f"Cluster {c} (n={(labels==c).sum():,})")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set(title=f"Residual clusters ({best}, relative residuals, k={k})", xlabel="Year",
           ylabel="Mean (actual − forecast) / (1 + 2013–15 mean)"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.out / "fig_residual_cluster_profiles.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in sorted(set(labels)):
        m = labels == c; ax.scatter(E[m, 0], E[m, 1], s=7, alpha=.45, label=f"Cluster {c}")
    ax.set(title="PCA of relative residual profiles, 2016–2024", xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"); ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(args.out / "fig_residual_pca.png", dpi=170); plt.close(fig)

    meta = dict(experiment="fixed-origin annual forecast", origin=ORIGIN_TEST, test_years=TEST_YEARS,
                train_origins=TRAIN_ORIGINS, dev_rows=int(len(dev)), test_rows=int(len(test)), postcodes=int(a.shape[0]),
                target="log1p(annual installations); metrics on raw scale, predictions clipped at 0",
                features=NUM_FEATURES + CAT_FEATURES, validation="expanding-origin folds: origin<=2012→2013, origin<=2013→2014",
                tuning={n: ("GridSearchCV" if specs[n][1] == "grid" else "BayesSearchCV n_iter=12") for n in specs},
                best_params=best_params, cv_log_mae=cv_scores, best_model=best,
                residual_scaling="(actual − forecast) / max(mean annual count 2013–2015, 10), winsorized at 0.5/99.5%; postcodes with <100 installations through 2015 excluded from clustering",
                clustered_postcodes=int(len(prof)),
                cluster_k=k, pca_variance_explained=[float(x) for x in pca.explained_variance_ratio_],
                seed=SEED, elapsed_seconds=round(time.monotonic() - t0, 1))
    (args.out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(overall.round(3).to_string(index=False)); print(byyear.pivot(index="year", columns="model", values="MAE").round(2))
    print(summ.round(3)); print(state_mix); print("elapsed", meta["elapsed_seconds"], "s")


if __name__ == "__main__":
    main()
