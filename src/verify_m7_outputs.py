#!/usr/bin/env python3
"""Verify Solar Uninterrupted Module 7 Experiment 3 outputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m7" / "adoption_growth"

POSTCODE = "Small Unit Installation Postcode"


required = [
    "holdout_model_metrics.csv",
    "holdout_predictions_2024.csv",
    "census_context_ablation.csv",
    "sensitivity_min_100_dwellings.csv",
    "mlflow_run_index.csv",
    "run_metadata.json",
    "fig_m7_holdout_mae_comparison.png",
    "fig_m7_actual_vs_predicted.png",
    "fig_m7_census_ablation.png",
]

for name in required:
    p = OUT / name
    assert p.exists(), f"missing {p}"
    assert p.stat().st_size > 0, f"empty {p}"

metrics = pd.read_csv(
    OUT / "holdout_model_metrics.csv"
)

pred = pd.read_csv(
    OUT / "holdout_predictions_2024.csv",
    dtype={POSTCODE: str},
)

ablation = pd.read_csv(
    OUT / "census_context_ablation.csv"
)

sensitivity = pd.read_csv(
    OUT / "sensitivity_min_100_dwellings.csv"
)

runs = pd.read_csv(
    OUT / "mlflow_run_index.csv"
)

meta = json.loads(
    (
        OUT / "run_metadata.json"
    ).read_text()
)

assert meta["train_origin"] == 2021
assert meta["validation_origin"] == 2022
assert meta["test_origin"] == 2023
assert meta["holdout_target_year"] == 2024

assert len(pred) > 2400
assert pred[POSTCODE].is_unique
assert pred["origin"].eq(2023).all()
assert pred["target_year"].eq(2024).all()
assert pred["dwellings_opd_2021"].ge(50).all()
assert pred["actual"].ge(0).all()

expected_feature_sets = {
    "baseline",
    "history_only",
    "history_plus_census",
}

assert expected_feature_sets.issubset(
    set(metrics["feature_set"])
)

for c in [
    "MAE",
    "RMSE",
    "MedAE",
    "R2",
    "MAE_pct_points",
]:
    assert np.isfinite(
        metrics[c]
    ).all(), f"non-finite {c}"

assert metrics["MAE"].ge(0).all()
assert metrics["RMSE"].ge(0).all()
assert metrics["MedAE"].ge(0).all()

expected_models = {
    "Ridge",
    "Random forest",
    "XGBoost",
    "RF-XGB blend",
}

for feature_set in [
    "history_only",
    "history_plus_census",
]:
    got = set(
        metrics.loc[
            metrics["feature_set"].eq(
                feature_set
            ),
            "model",
        ]
    )
    assert expected_models == got

baseline_models = set(
    metrics.loc[
        metrics["feature_set"].eq(
            "baseline"
        ),
        "model",
    ]
)

assert baseline_models == {
    "Previous year",
    "Mean previous 2 years",
}

assert set(ablation["model"]) == expected_models

assert sensitivity["min_dwellings"].eq(
    100
).all()

assert set(
    sensitivity["model"]
) == {
    "Previous year",
    "Mean previous 2 years",
    "Ridge",
    "Random forest",
    "XGBoost",
    "RF-XGB blend",
}

assert len(runs) >= 10
assert runs["run_id"].notna().all()
assert runs["run_id"].astype(str).str.len().gt(
    10
).all()

assert {
    "M6 Experiment 1",
    "M6 Experiment 2",
    "M7 Experiment 3",
}.issubset(
    set(runs["experiment"])
)

for figure in [
    "fig_m7_holdout_mae_comparison.png",
    "fig_m7_actual_vs_predicted.png",
    "fig_m7_census_ablation.png",
]:
    assert (
        OUT / figure
    ).stat().st_size > 10_000

print(
    "PASS chronological 2021→2022 / "
    "2022→2023 / 2023→2024 design"
)

print(
    "PASS primary sample, denominator guard, "
    "and unique 2024 holdout postcodes"
)

print(
    "PASS Ridge, Random Forest, XGBoost, "
    "blend, and naive baselines"
)

print(
    "PASS Census-context ablation and "
    ">=100-dwelling sensitivity analysis"
)

print(
    "PASS MLflow run index includes "
    "M6 Experiments 1–2 and M7 Experiment 3"
)

print(
    "PASS three rendered M7 analytical figures"
)

print("ALL M7 CHECKS PASSED")
