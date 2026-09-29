# Experiment 1 — one-month-ahead count backtest

Shashwat Bajaj · BDA 602 Solar Uninterrupted

This experiment fits model parameters on 2008–2015 monthly postcode installation counts and evaluates January 2016–December 2024. For a target month `t`, lag features may use actual observed installation counts through `t-1`; therefore this is an updated-history one-month-ahead backtest, not a fixed-origin multi-year forecast.

Primary models: Ridge Regression and Random Forest Regression. Persistence baselines: previous month and same month one year earlier. Tuning uses two expanding full-calendar-year validation folds (2014 and 2015). Residual profiles from the tuned Random Forest are explored with PCA and K-Means.

Run from the repository root:

```bash
python src/run_m5_count_analysis.py
python src/verify_m5_outputs.py
```

The canonical input is `data/processed/cer_solar_panel_long.csv`, loaded through `src.build_panel.load_panel()` so postcode strings and the metadata sidecar are preserved.

These outputs model **installation counts**, not household adoption rates. Experiment 2 in `src/fixed_origin_forecast.py` addresses the separate fixed-origin design.
