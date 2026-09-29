#!/usr/bin/env python3
"""Fast integrity checks for Shashwat's Module 6 outputs."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M6 = ROOT / "results" / "m6"

for folder in ["experiment1_expanded", "experiment2_expanded", "abs_cluster_profile"]:
    assert (M6 / folder).is_dir(), f"missing {folder}"

m1 = json.loads((M6 / "experiment1_expanded" / "run_metadata.json").read_text())
assert "BayesSearchCV" in m1["random_forest_search"]
assert m1["rows_train"] == 269664 and m1["rows_holdout"] == 303480
assert m1["postcodes"] == 2810

e1 = pd.read_csv(M6 / "experiment1_expanded" / "holdout_model_metrics.csv")
assert set(["Previous month", "Same month last year", "Ridge", "Random forest"]).issubset(set(e1.model))
assert (e1[["MAE", "RMSE"]].to_numpy() >= 0).all()
assert len(pd.read_csv(M6 / "experiment1_expanded" / "ridge_search.csv")) == 11
assert len(pd.read_csv(M6 / "experiment1_expanded" / "random_forest_search.csv")) >= 12
print("PASS Experiment 1 expanded search")

m2 = json.loads((M6 / "experiment2_expanded" / "run_metadata.json").read_text())
assert m2["dev_rows"] == 126450 and m2["test_rows"] == 25290
assert "BayesSearchCV" in m2["searches"]["Random forest"]
assert "BayesSearchCV" in m2["searches"]["XGBoost"]
e2 = pd.read_csv(M6 / "experiment2_expanded" / "holdout_model_metrics.csv")
expected = {"Persistence (2015 count)", "Mean 2013–2015", "Ridge", "Decision tree", "Random forest", "XGBoost"}
assert expected.issubset(set(e2.model))
assert (e2[["MAE", "RMSE"]].to_numpy() >= 0).all()
print("PASS Experiment 2 expanded search")

am = json.loads((M6 / "abs_cluster_profile" / "run_metadata.json").read_text())
assert "descriptive" in am["analysis"]
assert "not causal" in am["interpretation_rule"]
eff = pd.read_csv(M6 / "abs_cluster_profile" / "cluster_feature_effects.csv")
assert len(eff) >= 5
assert (M6 / "abs_cluster_profile" / "fig_abs_cluster_standardized_differences.png").stat().st_size > 10000
print("PASS ABS residual-cluster profile")

assert (ROOT / "docs" / "M6_Shashwat_Modeling_Update.md").stat().st_size > 2500
print("PASS copy-ready Module 6 prose")
print("ALL M6 CHECKS PASSED")
