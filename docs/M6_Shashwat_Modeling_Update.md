# Solar Uninterrupted — Module 6 Modeling Update

**Prepared by:** Shashwat Bajaj, Modeling Lead  
**Purpose:** copy-ready material for Abraham's combined Module 6 draft.

## 1. Expanded hyperparameter optimization

Module 6 expands the searches used in the first draft while preserving the same chronological validation designs. Experiment 1 retains the two expanding full-year folds (training through 2013 and validating on 2014; then training through 2014 and validating on 2015). Ridge now uses an 11-value alpha grid, and Random Forest uses BayesSearchCV across a broader depth, minimum-leaf, feature-subsampling, and row-subsampling space with 150 trees per fit and 16 Bayesian iterations. No held-out 2016–2024 observations are used for tuning.

Experiment 2 retains the two expanding-origin folds (origins through 2012 validated on origin 2013, then origins through 2013 validated on origin 2014). Ridge and Decision Tree use larger grids; Random Forest and XGBoost use BayesSearchCV over broader spaces, with 200 trees and 300 boosting rounds respectively and 20 Bayesian iterations. The final 2015-origin forecasts for 2016–2024 remain a single held-out evaluation.

### Selected parameters

- Experiment 1 Ridge: `{'model__alpha': 1000}`; development-fold MAE 2.305.
- Experiment 1 Random Forest: `{'max_depth': 15, 'max_features': 0.3, 'max_samples': 0.9, 'min_samples_leaf': 8}`; development-fold MAE 2.003.
- Experiment 2 best parameters: `{'Ridge': {'m__alpha': 1000}, 'Decision tree': {'m__max_depth': 6, 'm__min_samples_leaf': 150}, 'Random forest': {'m__max_depth': 24, 'm__max_features': 0.2, 'm__min_samples_leaf': 3}, 'XGBoost': {'m__colsample_bytree': 0.5697583222478744, 'm__learning_rate': 0.01, 'm__max_depth': 8, 'm__min_child_weight': 29, 'm__subsample': 0.9907769799789264}}`.

## 2. Experiment 1 — expanded-search results

The expanded search leaves the Experiment 1 forecasting interpretation unchanged: model weights are estimated from 2008–2015, while each held-out month can use actual installation history through the preceding month. The best learned model by held-out MAE is **Random forest**. Ridge records MAE 2.901, RMSE 5.933, and R² 0.868; Random Forest records MAE 2.648, RMSE 5.957, and R² 0.867. Previous-month persistence has MAE 3.027.

Compared with the Module 5 search, Ridge: M5 MAE 2.906 → M6 MAE 2.901 (0.2% lower); Random forest: M5 MAE 2.672 → M6 MAE 2.648 (0.9% lower).

The important methodological point remains that strong one-month-ahead performance does not imply that a model trained through 2015 can forecast many years into a different market regime. Experiment 1 measures short-horizon tracking when recent observations are available.

## 3. Experiment 2 — expanded-search results

With every feature frozen at the end of 2015, the best learned model by held-out MAE is **Decision tree**, at MAE 54.1 installations per postcode-year. The 2013–2015 mean baseline has MAE 46.8. Therefore, the expanded tuning **does not** change the central fixed-origin finding: the best learned model still does not beat the simple three-year mean on the 2016–2024 holdout.

Relative to the Module 5 search, Ridge: 105.7 → 102.7 MAE; Decision tree: 54.1 → 54.1 MAE; Random forest: 56.9 → 58.6 MAE; XGBoost: 66.8 → 63.9 MAE.

This result is useful precisely because the wider searches reduce the chance that the first-draft conclusion was an artifact of an unusually narrow hyperparameter grid. Any remaining performance gap should still be interpreted in the context of a structural market shift rather than as proof that one algorithm class is universally weak.

## 4. ABS profiling of residual clusters

The 2021 Census join is used only for **post-hoc descriptive profiling** of the existing Experiment 2 residual clusters. Census variables are a 2021 snapshot and therefore are not treated as information available to the 2015-origin forecast. The December 2024 adoption-rate snapshot also uses the static 2021 occupied-private-dwelling denominator, so these comparisons are descriptive rather than causal or historically contemporaneous.

The largest standardized differences between the two residual clusters are:

- `Median_rent_weekly`: standardized mean difference (Cluster 1 − Cluster 0) = +0.94; cluster medians 320 vs. 400.
- `Median_mortgage_repay_monthly`: standardized mean difference (Cluster 1 − Cluster 0) = +0.94; cluster medians 1.62e+03 vs. 2.15e+03.
- `Median_tot_hhd_inc_weekly`: standardized mean difference (Cluster 1 − Cluster 0) = +0.69; cluster medians 1.5e+03 vs. 1.94e+03.
- `pop_density_per_sqkm`: standardized mean difference (Cluster 1 − Cluster 0) = +0.61; cluster medians 54 vs. 1.12e+03.

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
