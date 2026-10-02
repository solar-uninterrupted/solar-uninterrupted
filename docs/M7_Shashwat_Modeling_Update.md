# Solar Uninterrupted — Module 7 Modeling Update

**Prepared by:** Shashwat Bajaj, Modeling Lead
**Purpose:** copy-ready Methodology and Analysis & Results material for the Module 7 near-final paper.

## 1. Experiment 3 — one-year-ahead adoption-growth forecasting

Module 7 adds a third supervised experiment to the two installation-count forecasting designs reported previously. Experiment 3 predicts the next calendar year's increase in postcode-level rooftop-solar adoption rate:

**annual installations / 2021 occupied private dwellings**

The target is an annual increment rather than cumulative adoption rate. This avoids making the prediction task largely a restatement of the previous cumulative adoption level.

The experiment uses a chronological three-origin target split:

- origin 2021 → target 2022: tuning-training sample
- origin 2022 → target 2023: chronological validation sample
- origin 2023 → target 2024: untouched final holdout

The 2024 holdout is never used for hyperparameter selection or ensemble weighting.

The primary sample excludes ABS small-population areas and postcodes with fewer than 50 occupied private dwellings. This leaves 2,476 postcodes in each origin and 2,476 postcodes in the final 2024 holdout. A second analysis raises the denominator threshold to 100 dwellings as a sensitivity test.

## 2. Feature sets and models

Two feature sets are compared.

### History-only

The history-only specification uses recent adoption and installation information available by the forecast origin, including:

- annual adoption-rate increments over the previous three years
- two- and three-year adoption-growth averages
- recent adoption-rate trend
- cumulative adoption through the forecast origin
- previous-year installations
- three-year mean installations
- occupied-dwelling scale
- years with observed solar-installation activity

### History plus Census context

The second specification adds the fixed ABS 2021 Census snapshot:

- median age
- median household income
- average household size
- median mortgage repayment
- median rent
- population density
- detached, semi-detached, and apartment shares
- unoccupied dwelling share
- owner-occupied and rented shares
- state

The Census features describe the population and housing structure measured at the 2021 Census, whose reference date predates all three Experiment 3 target years. Most 2021 Census topics were publicly released on **28 June 2022**. Accordingly, the 2021→2022 development row is retrospective with respect to publication availability, while the 2022→2023 validation and 2023→2024 holdout are genuine out-of-time forecasts with Census data available by the forecasting origin. The 2021 Census variables are never inserted into the genuine end-2015 fixed-origin Experiment 2 forecast.

Source: Australian Bureau of Statistics, [2021 Census product release guide](https://www.abs.gov.au/census/guide-census-data/2021-census-product-release-guide).

The supervised models are:

- Ridge regression
- Random Forest
- XGBoost
- a validation-selected Random Forest/XGBoost weighted blend

Two naive comparators are retained:

- previous year's adoption-rate increment
- mean of the previous two annual adoption-rate increments

Hyperparameters are selected using the chronological 2022→2023 validation step. The final selected models are then refit on both development origins and evaluated once on the untouched 2024 holdout.

Evaluation uses MAE, RMSE, median absolute error, and R².

## 3. Experiment 3 holdout results

The strongest naive comparator is the mean of the previous two years, with:

- MAE: 0.01148
- RMSE: 0.02421
- median absolute error: 0.00599
- R²: 0.397

The lowest holdout MAE is produced by the **history-plus-Census Random Forest/XGBoost blend**:

- MAE: 0.01120
- RMSE: 0.02442
- median absolute error: 0.00656
- R²: 0.387

This is approximately a **2.4% reduction in MAE** relative to the strongest naive baseline.

The gain should be interpreted carefully. The blend achieves the lowest mean absolute error, but it does not dominate every metric. For example, the previous-year baseline has RMSE 0.02313 and R² 0.450, which are slightly better than the blend on the primary sample.

The appropriate conclusion is therefore that the ensemble modestly reduces average absolute forecast error, not that it is universally superior.

### Primary-sample comparison

| Feature set | Model | MAE | RMSE | MedAE | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Previous year | 0.01196 | 0.02313 | 0.00658 | 0.450 |
| Baseline | Mean previous 2 years | 0.01148 | 0.02421 | 0.00599 | 0.397 |
| History only | Ridge | 0.01171 | 0.02330 | 0.00711 | 0.442 |
| History only | Random Forest | 0.01126 | 0.02480 | 0.00643 | 0.368 |
| History only | XGBoost | 0.01157 | 0.02559 | 0.00682 | 0.327 |
| History only | RF/XGB blend | 0.01126 | 0.02480 | 0.00643 | 0.368 |
| History + Census | Ridge | 0.01143 | 0.02323 | 0.00698 | 0.445 |
| History + Census | Random Forest | 0.01123 | 0.02456 | 0.00660 | 0.380 |
| History + Census | XGBoost | 0.01141 | 0.02393 | 0.00654 | 0.411 |
| History + Census | RF/XGB blend | **0.01120** | 0.02442 | 0.00656 | 0.387 |

## 4. Census-context ablation

Adding Census context lowers holdout MAE for each learned specification:

| Model | History-only MAE | History + Census MAE | MAE change |
| --- | ---: | ---: | ---: |
| Ridge | 0.01171 | 0.01143 | -0.00028 |
| Random Forest | 0.01126 | 0.01123 | -0.00003 |
| XGBoost | 0.01157 | 0.01141 | -0.00017 |
| RF/XGB blend | 0.01126 | 0.01120 | -0.00006 |

The largest improvement occurs for Ridge. The tree ensembles change less. Recent adoption history therefore carries most of the short-horizon predictive signal, while the Census snapshot contributes incremental structural information.

This is an ablation result rather than evidence that the Census variables cause later adoption.

## 5. Denominator sensitivity

The adoption-rate denominator is the number of occupied private dwellings measured in the 2021 Census. Small denominators can amplify installation-count noise and the effects of commercial systems, replacements, or other departures from a literal one-system-per-household interpretation.

For this reason, the analysis is repeated after requiring at least 100 occupied private dwellings.

On the ≥100-dwelling sensitivity sample:

| Model | MAE | RMSE | MedAE | R² |
| --- | ---: | ---: | ---: | ---: |
| Previous year | 0.01104 | 0.02141 | 0.00621 | 0.215 |
| Mean previous 2 years | 0.01034 | 0.02007 | 0.00549 | 0.310 |
| Ridge | 0.01026 | 0.01778 | 0.00630 | 0.459 |
| Random Forest | 0.01010 | 0.01739 | 0.00616 | 0.482 |
| XGBoost | 0.01019 | 0.01815 | 0.00601 | 0.436 |
| RF/XGB blend | **0.01008** | **0.01737** | 0.00613 | **0.484** |

The blend improves MAE by approximately **2.6%** relative to the strongest naive baseline on this sensitivity sample, while also improving RMSE and R².

The stronger result after removing the smallest denominators indicates that denominator instability is an important source of postcode-level forecast noise.

## 6. Connection to the residual-cluster analysis

Experiment 2 identified 393 active postcodes whose installations systematically outran the fixed-origin forecast.

The ABS profile shows that this group is:

- higher-income
- denser
- younger
- more apartment-oriented
- less detached-house dominated

Its median 2024 cumulative adoption rate is approximately **0.41**, compared with approximately **0.56** in the other residual cluster.

This combination is consistent with later-starting urban areas having more remaining adoption headroom. It also complements the strong ACT representation identified in the geographic cluster analysis.

These patterns remain **descriptive associations only**. The 2021 Census features were not information available to the end-2015 forecast and do not establish a causal explanation for the forecast errors.

## 7. Cross-experiment interpretation

The three experiments answer different forecasting questions.

### Experiment 1 — one-month-ahead installation counts

Recent observed history is highly informative. The expanded Random Forest reaches MAE 2.648 compared with 3.027 for previous-month persistence.

### Experiment 2 — fixed-origin forecast from end-2015

Wider hyperparameter tuning does not overturn the long-horizon result. The 2013–2015 mean baseline remains stronger than the learned models over the 2016–2024 holdout.

This indicates that tuning within the pre-2016 market regime cannot supply information about the later market shift.

### Experiment 3 — one-year-ahead adoption growth

Recent history again carries most of the predictive signal. Census context modestly improves MAE, and the denominator sensitivity shows that very small areas introduce additional instability.

Taken together, the experiments support a practical conclusion: **short-horizon forecasts can perform well when recent observations are available, while long-horizon performance deteriorates when the underlying market regime changes. Structural demographic context can refine near-term forecasts, but it does not substitute for model updating.**

## 8. MLflow and reproducibility

Module 7 uses MLflow 3 with a local SQLite backend.

Experiment 3 runs directly record:

- feature set
- chronological train/validation/holdout origins
- sample restrictions
- selected model parameters
- ensemble-selection information
- validation MAE
- final holdout metrics

For cross-experiment comparison, verified summaries of completed M6 Experiments 1 and 2 are also registered in MLflow.

Those M6 entries should be described as **retrospectively registered verified summaries**. The original M6 model-training executions themselves were not run under MLflow.

The local `mlflow.db` is gitignored. Committed run identifiers are stored in:

`results/m7/adoption_growth/mlflow_run_index.csv`

## 9. Recommended analytical visuals

The Module 7 paper/presentation can use:

1. `fig_m7_holdout_mae_comparison.png` — supervised models and naive baselines.
2. `fig_m7_actual_vs_predicted.png` — actual versus predicted 2024 adoption growth.
3. `fig_m7_census_ablation.png` — direct effect of adding Census context.

The existing Experiment 2 forecast-horizon and residual-cluster figures should remain in the broader paper because they show the long-horizon degradation and unsupervised error structure.

## 10. Limitations to retain

- Adoption rate uses a static 2021 occupied-private-dwelling denominator across multiple years.
- The CER installation data do not perfectly separate household systems, commercial systems, replacements, or every registration-timing effect.
- The 2021 Census snapshot predates all Experiment 3 target years, but public release occurred on 28 June 2022; the 2021→2022 development row is therefore retrospective with respect to publication availability, while the later validation and holdout are out-of-time.
- The primary-sample MAE gain is modest and does not dominate every metric.
- Socioeconomic and housing characteristics associated with residual clusters are descriptive, not causal.

## 11. About the Author — Shashwat Bajaj

Shashwat Bajaj is a graduate student pursuing a Master of Science in Big Data Analytics at San Diego State University. His academic interests include machine learning, predictive modeling, and applying data analytics to real-world problems. As Modeling Lead for Solar Uninterrupted, he developed and evaluated the supervised and unsupervised modeling workflows, implemented time-ordered validation and experiment tracking, and analyzed residual and adoption-growth patterns across Australian postcodes.

**Contact:** sbajaj5920@sdsu.edu
