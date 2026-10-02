# Module 7 — Experiment 3: Adoption-Growth Forecasting

**Owner:** Shashwat Bajaj, Modeling Lead

## Research question

Can the next year's increase in postcode-level rooftop-solar adoption rate be forecast from recent adoption history, and does a fixed 2021 Census context improve that forecast?

## Target

Annual adoption-rate increment:

`annual installations / 2021 occupied private dwellings`

This is an annual increment, not cumulative adoption rate.

## Chronological design

- origin 2021 → target 2022: tuning-training
- origin 2022 → target 2023: chronological validation
- origin 2023 → target 2024: untouched holdout

No 2024 target values are used for model selection.

## Primary sample

- `small_pop_flag == 0`
- at least 50 occupied private dwellings
- 2,476 postcodes in the holdout

A ≥100-dwelling sensitivity analysis is also reported.

## Models

- Ridge
- Random Forest
- XGBoost
- validation-selected Random Forest/XGBoost weighted blend
- previous-year naive baseline
- two-year-mean naive baseline

## Feature sets

- `history_only`
- `history_plus_census`

The Census reference date predates all Experiment 3 target years. Because most 2021 Census topics were publicly released on 28 June 2022, the 2021→2022 development row is retrospective with respect to publication availability, while the 2022→2023 validation and 2023→2024 holdout are out-of-time forecasts with Census data available by the forecasting origin. The Census branch does not alter the end-2015 Experiment 2 forecast.

## Reproduce

From the repository root:

~~~bash
python src/m7_adoption_ensemble.py
python src/verify_m7_outputs.py
~~~

The run creates a local gitignored `mlflow.db` SQLite database.

Inspect the MLflow experiment locally with:

~~~bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
~~~

## Main result

Best 2024 holdout MAE:

**history_plus_census / RF-XGB blend = 0.01120**

Best naive holdout MAE:

**Mean previous 2 years = 0.01148**

This is approximately a 2.4% MAE improvement. It is a modest gain and does not dominate every evaluation metric.

## Sensitivity result

After requiring at least 100 occupied private dwellings, the RF/XGB blend records:

- MAE 0.01008
- RMSE 0.01737
- R² 0.484

The two-year-mean baseline records MAE 0.01034.

## Output files

- `holdout_model_metrics.csv`
- `holdout_predictions_2024.csv`
- `census_context_ablation.csv`
- `sensitivity_min_100_dwellings.csv`
- `mlflow_run_index.csv`
- `run_metadata.json`
- `search/`
- `fig_m7_holdout_mae_comparison.png`
- `fig_m7_actual_vs_predicted.png`
- `fig_m7_census_ablation.png`

## Interpretation boundary

The 2021 Census variables provide structural context for Experiment 3. Their reference date predates all three target years, but public release occurred during 2022, so only the later validation and holdout are described as out-of-time with respect to Census availability. The Census variables are not treated as information available to the end-2015 Experiment 2 forecast, and observed Census/residual associations are not interpreted causally.
