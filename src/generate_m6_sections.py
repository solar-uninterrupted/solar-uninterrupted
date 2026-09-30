#!/usr/bin/env python3
"""Generate copy-ready Shashwat Module 6 prose from completed M6 outputs."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M6 = ROOT / "results" / "m6"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def f3(x):
    return f"{float(x):.3f}"


def pct_change(old, new):
    if old == 0:
        return "n/a"
    v = 100 * (new-old) / old
    return f"{abs(v):.1f}% {'lower' if v < 0 else 'higher'}"


e1 = pd.read_csv(M6 / "experiment1_expanded" / "holdout_model_metrics.csv").set_index("model")
e1meta = json.loads((M6 / "experiment1_expanded" / "run_metadata.json").read_text())
e2 = pd.read_csv(M6 / "experiment2_expanded" / "holdout_model_metrics.csv").set_index("model")
e2meta = json.loads((M6 / "experiment2_expanded" / "run_metadata.json").read_text())
effects = pd.read_csv(M6 / "abs_cluster_profile" / "cluster_feature_effects.csv")
absmeta = json.loads((M6 / "abs_cluster_profile" / "run_metadata.json").read_text())

old1_path = M6 / "experiment1_expanded" / "m5_vs_m6_metrics.csv"
old2_path = M6 / "experiment2_expanded" / "m5_vs_m6_metrics.csv"
old1 = pd.read_csv(old1_path).set_index("model") if old1_path.exists() else None
old2 = pd.read_csv(old2_path).set_index("model") if old2_path.exists() else None

best_e1 = e1.loc[["Ridge", "Random forest"]].sort_values("MAE").index[0]
best_e2 = e2.loc[["Ridge", "Decision tree", "Random forest", "XGBoost"]].sort_values("MAE").index[0]
naive2 = float(e2.loc["Mean 2013–2015", "MAE"])
best2_mae = float(e2.loc[best_e2, "MAE"])
beats_naive = best2_mae < naive2

abs_lines = []
for _, r in effects.head(4).iterrows():
    abs_lines.append(
        f"- `{r['feature']}`: standardized mean difference (Cluster 1 − Cluster 0) "
        f"= {float(r['standardized_mean_difference_c1_minus_c0']):+.2f}; "
        f"cluster medians {float(r['cluster0_median']):.3g} vs. {float(r['cluster1_median']):.3g}."
    )

m5cmp1 = ""
if old1 is not None:
    rows=[]
    for model in ["Ridge", "Random forest"]:
        if model in old1.index:
            rows.append(
                f"{model}: M5 MAE {old1.loc[model,'m5_mae']:.3f} → M6 MAE {old1.loc[model,'m6_mae']:.3f} "
                f"({pct_change(old1.loc[model,'m5_mae'], old1.loc[model,'m6_mae'])})"
            )
    m5cmp1 = "; ".join(rows)

m5cmp2 = ""
if old2 is not None:
    rows=[]
    for model in ["Ridge", "Decision tree", "Random forest", "XGBoost"]:
        if model in old2.index:
            rows.append(
                f"{model}: {old2.loc[model,'m5_mae']:.1f} → {old2.loc[model,'m6_mae']:.1f} MAE"
            )
    m5cmp2 = "; ".join(rows)

text = f"""# Solar Uninterrupted — Module 6 Modeling Update

**Prepared by:** Shashwat Bajaj, Modeling Lead  
**Purpose:** copy-ready material for Abraham's combined Module 6 draft.

## 1. Expanded hyperparameter optimization

Module 6 expands the searches used in the first draft while preserving the same chronological validation designs. Experiment 1 retains the two expanding full-year folds (training through 2013 and validating on 2014; then training through 2014 and validating on 2015). Ridge now uses an 11-value alpha grid, and Random Forest uses BayesSearchCV across a broader depth, minimum-leaf, feature-subsampling, and row-subsampling space with 150 trees per fit and {e1meta['random_forest_search'].split('n_iter=')[-1].split(';')[0]} Bayesian iterations. No held-out 2016–2024 observations are used for tuning.

Experiment 2 retains the two expanding-origin folds (origins through 2012 validated on origin 2013, then origins through 2013 validated on origin 2014). Ridge and Decision Tree use larger grids; Random Forest and XGBoost use BayesSearchCV over broader spaces, with 200 trees and 300 boosting rounds respectively and {e2meta['searches']['Random forest'].split('=')[-1]} Bayesian iterations. The final 2015-origin forecasts for 2016–2024 remain a single held-out evaluation.

### Selected parameters

- Experiment 1 Ridge: `{e1meta['ridge_best_params']}`; development-fold MAE {e1meta['ridge_cv_mae']:.3f}.
- Experiment 1 Random Forest: `{e1meta['rf_best_params']}`; development-fold MAE {e1meta['rf_cv_mae']:.3f}.
- Experiment 2 best parameters: `{e2meta['best_params']}`.

## 2. Experiment 1 — expanded-search results

The expanded search leaves the Experiment 1 forecasting interpretation unchanged: model weights are estimated from 2008–2015, while each held-out month can use actual installation history through the preceding month. The best learned model by held-out MAE is **{best_e1}**. Ridge records MAE {f3(e1.loc['Ridge','MAE'])}, RMSE {f3(e1.loc['Ridge','RMSE'])}, and R² {f3(e1.loc['Ridge','R2'])}; Random Forest records MAE {f3(e1.loc['Random forest','MAE'])}, RMSE {f3(e1.loc['Random forest','RMSE'])}, and R² {f3(e1.loc['Random forest','R2'])}. Previous-month persistence has MAE {f3(e1.loc['Previous month','MAE'])}.

{('Compared with the Module 5 search, ' + m5cmp1 + '.') if m5cmp1 else ''}

The important methodological point remains that strong one-month-ahead performance does not imply that a model trained through 2015 can forecast many years into a different market regime. Experiment 1 measures short-horizon tracking when recent observations are available.

## 3. Experiment 2 — expanded-search results

With every feature frozen at the end of 2015, the best learned model by held-out MAE is **{best_e2}**, at MAE {best2_mae:.1f} installations per postcode-year. The 2013–2015 mean baseline has MAE {naive2:.1f}. Therefore, the expanded tuning **{'does' if beats_naive else 'does not'}** change the central fixed-origin finding: the best learned model {'beats' if beats_naive else 'still does not beat'} the simple three-year mean on the 2016–2024 holdout.

{('Relative to the Module 5 search, ' + m5cmp2 + '.') if m5cmp2 else ''}

This result is useful precisely because the wider searches reduce the chance that the first-draft conclusion was an artifact of an unusually narrow hyperparameter grid. Any remaining performance gap should still be interpreted in the context of a structural market shift rather than as proof that one algorithm class is universally weak.

## 4. ABS profiling of residual clusters

The 2021 Census join is used only for **post-hoc descriptive profiling** of the existing Experiment 2 residual clusters. Census variables are a 2021 snapshot and therefore are not treated as information available to the 2015-origin forecast. The December 2024 adoption-rate snapshot also uses the static 2021 occupied-private-dwelling denominator, so these comparisons are descriptive rather than causal or historically contemporaneous.

The largest standardized differences between the two residual clusters are:

{chr(10).join(abs_lines)}

These differences should be reported as associations that help characterize where the fixed-origin forecast failed. They do not establish that income, dwelling structure, tenure, density, or any other Census characteristic caused the forecast error.

## 5. Reproducibility / GitHub note

The M6 analysis is committed separately from the Module 5 Experiment 1 handoff. Each run records the validation structure, search method, selected parameters, random seed, and output tables under `results/m6/`. This preserves the project history requested for Module 6 and makes the expanded-search results independently reviewable.

## 6. Files for integration

- `results/m6/experiment1_expanded/holdout_model_metrics.csv`
- `results/m6/experiment1_expanded/*_search.csv`
- `results/m6/experiment2_expanded/holdout_model_metrics.csv`
- `results/m6/experiment2_expanded/*_search.csv`
- `results/m6/abs_cluster_profile/cluster_feature_summary.csv`
- `results/m6/abs_cluster_profile/cluster_feature_effects.csv`
- `results/m6/abs_cluster_profile/fig_abs_cluster_standardized_differences.png`
- `results/m6/m6_summary.json`

### Caution to retain in the combined paper

The CER count experiments and the ABS adoption-rate / Census profiling answer different questions. Experiment 2 remains the genuine out-of-time fixed-origin forecasting test; 2021 Census features are used retrospectively to interpret residual groups and must not be described as predictors that were available in 2015.
"""

out = DOCS / "M6_Shashwat_Modeling_Update.md"
out.write_text(text)
print(out)
