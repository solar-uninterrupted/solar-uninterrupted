# Solar Uninterrupted — Prediction and Reality in Australian Rooftop Solar

BDA 602 Applied Machine Learning Engineering · SDSU Global Campus · Fall 2026
Abraham Cho (Group Leader, Data Lead) · Shashwat Bajaj (Modeling Lead)

Forecasting rooftop-solar adoption across 2,811 Australian postcodes, training on
2001–2015 and predicting 2016–2026, then clustering each postcode's residual series
to find where — and how — the forecast systematically fails.

## Repository layout

```
data/raw/          CER source files (as downloaded; do not edit)
data/raw/abs/      ABS 2021 Census GCP Postal Area tables G01, G02, G36, G37 + POA area/state
data/processed/    cer_solar_panel_long.csv + .meta.json sidecar (built by src/build_panel.py)
                   abs_poa_features.csv, cer_abs_panel_long.csv.gz, abs_join_report.json (built by src/join_abs.py)
src/build_panel.py Reproducible reshape from wide to long; also exposes load_panel()
src/join_abs.py    CER–ABS join: adoption-rate target + Census features; exposes load_joined(), load_features()
src/fixed_origin_forecast.py  Experiment 2: fixed-origin annual forecast from end-2015, 4 tuned models, residual clustering
results/experiment2/  Experiment 2 outputs: metrics, tuning tables, predictions, clusters, figures, run_metadata.json
notebooks/         EDA and modeling notebooks
docs/              Submitted deliverables (M1–M4) and the ABS data instructions
```

## Data

### Clean Energy Regulator — small-scale installation postcode data
Source: https://cer.gov.au/markets/reports-and-data/small-scale-installation-postcode-data
(current as at 31 July 2026). Two wide CSVs, one column per month:

| File | Shape | Monthly column name |
|---|---|---|
| sgu-solar-installations-2001-to-2010.csv | 2,809 × 118 | `<Mon YYYY> - Installations Quantity` |
| sgu-solar-installations-2011-to-present-and-totals.csv | 2,811 × 190 | `<Mon YYYY> - Installation Quantity` |

### Processed panel — `data/processed/cer_solar_panel_long.csv`
854,310 rows × 3 columns: `Small Unit Installation Postcode` (string), `month` (YYYY-MM-DD),
`installations` (int). 2,811 postcodes, 304 months (Apr 2001 – Jul 2026), no missing values,
total 4,501,887 installations (reconciles with the CER published figure).

**Load it with the helper, not with a bare `pd.read_csv`:**

```python
from src.build_panel import load_panel
panel = load_panel("data/processed/cer_solar_panel_long.csv")
```

`load_panel` reads the `.meta.json` sidecar, forces postcode to string, parses `month`,
and asserts the row count. A plain `pd.read_csv` silently turns postcode `0800` (Darwin)
into `800` and the demographic join then fails without an error.

To rebuild from the raw files:

```
python src/build_panel.py --raw data/raw --out data/processed
```

### ABS 2021 Census join — `src/join_abs.py`
Source: ABS 2021 Census General Community Profile DataPack, Postal Areas, Australia (short header).
Only the four tables used are in `data/raw/abs/`; the full pack is in the group Drive folder.

```
python src/join_abs.py
```

Outputs in `data/processed/`:

| File | Grain | Contents |
|---|---|---|
| `abs_poa_features.csv` | 2,643 Postal Areas | population, medians (age, household income, rent, mortgage), household size, dwelling-structure shares, tenure shares, population density, `state_abs`, `small_pop_flag`, `cross_border_flag` |
| `cer_abs_panel_long.csv.gz` | 800,736 postcode-months | matched postcodes only: `installations`, `cum_installations`, `dwellings_opd_2021`, `adoption_rate`, `rate_gt_1_flag` |
| `abs_join_report.json` | — | match counts and the unmatched postcode list |

2,634 of 2,810 CER postcodes (excluding `0000`) match a Postal Area. The 176 unmatched are
PO-box and delivery-centre postcodes holding 2,179 installations (0.05%).
**Adoption rate** = cumulative installations since Apr 2001 ÷ 2021 occupied private dwellings
(G36 `OPDs_Tot_OPDs_Dwellings`). The denominator is static; 174 postcodes exceed 1 by Jul 2026
and are flagged, not dropped. Census features describe 2021, so using them to explain earlier
adoption is retrospective explanation, not forecasting.

```python
from src.join_abs import load_joined
df = load_joined("data/processed/cer_abs_panel_long.csv.gz",
                 features="data/processed/abs_poa_features.csv")
```

## Data-handling decisions

These are documented in the M2 planning summary and implemented or flagged as noted.

| Issue | What happens | Where |
|---|---|---|
| Leading zeros (48 postcodes) | Postcode read as string throughout | `build_panel.py`, sidecar |
| Schema drift between files | One regex parses both column-name variants | `build_panel.py` |
| Totals columns in 2011+ file | Dropped from the panel; used only to reconcile the total | `build_panel.py` |
| Registration lag (12 months) | Not applied in the panel. Truncate or lag-adjust the last ~12 months before modelling; 2026 months in particular undercount | modelling code |
| Privacy floor (cells < 10 modified) | Not applied. Add an indicator feature; do not treat small counts as exact | modelling code |
| No residential / commercial split | Not applied. Capacity threshold following APVI, with sensitivity reported | modelling code |
| Postcode `0000` | Present in the CER files as a placeholder; exclude from geographic joins | modelling code |

## Experiments

- **Experiment 1 — one-month-ahead backtest** (Shashwat, `code/run_m5_count_analysis.py` on branch `m5-modeling`): weights fitted 2008–2015, each 2016–2024 prediction uses actual counts through the prior month.
- **Experiment 2 — fixed-origin forecast** (`src/fixed_origin_forecast.py`): every 2016–2024 prediction uses only information available at end-2015; error reported by horizon; residual clustering on relative residuals. Run: `python src/fixed_origin_forecast.py --stage all` (or `--stage tune --model NAME` then `--stage fit` on a slow machine).

## Modelling plan (from the M4 paper outline)

- **Target:** postcode-level installation activity; adoption *rate* requires the ABS dwelling
  denominator (see `docs/ABS-data-instructions.md`), so the M5 first pass models installation
  counts and says so explicitly.
- **Split:** chronological. Train 2001–2015, evaluate 2016–2026 with metrics reported by
  prediction year. Tuning uses expanding-window folds inside 2001–2015 only.
- **Supervised:** ridge regression and random forest (primary); decision tree and XGBoost (additional).
- **Tuning:** GridSearchCV / RandomizedSearchCV; BayesSearchCV (scikit-optimize) for the tree ensembles.
- **Unsupervised:** PCA + K-Means on each postcode's out-of-sample residual series; hierarchical clustering as a check.
- **Metrics:** MAE, RMSE, R², plus a persistence baseline (last year's value) that every model must beat.
- **Leakage rule:** any installation-history feature attached to a row dated *t* uses only data before *t*.

## Environment

Python ≥ 3.10, pandas, numpy, scikit-learn, xgboost, scikit-optimize, mlflow, matplotlib, seaborn.

```
pip install -r requirements.txt
```
