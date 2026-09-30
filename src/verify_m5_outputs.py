#!/usr/bin/env python3
"""Independent on-disk checks for the provisional count-only Module 5 results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1]
res=ROOT/'results'/'experiment1'
meta=json.loads((res/'run_metadata.json').read_text())
assert meta['count_only'] is True and meta['geographic_map_completed'] is False
assert meta['rolling_one_month_ahead_uses_observed_post_2015_lags'] is True
assert meta['bayessearchcv_run'] is False
pred=pd.read_csv(res/'holdout_predictions.csv.gz',dtype={'postcode':str},parse_dates=['month'])
assert len(pred)==303480 and pred.postcode.nunique()==2810
assert pred.month.min()==pd.Timestamp('2016-01-01')
assert pred.month.max()==pd.Timestamp('2024-12-01')
assert not pred.duplicated(['postcode','month']).any()
assert pred.actual.sum()==2533061
assert pred.postcode.str.fullmatch(r'\d{4}').all()
for label in ['Previous month','Same month last year','Ridge','Random forest']:
    p=pred[label].to_numpy()
    y=pred.actual.to_numpy()
    assert np.isfinite(p).all() and (p>=0).all()
    reported=pd.read_csv(res/'holdout_model_metrics.csv').set_index('model').loc[label]
    assert abs(mean_absolute_error(y,p)-reported.MAE)<.00001
    assert abs(np.sqrt(mean_squared_error(y,p))-reported.RMSE)<.00001
    assert abs(r2_score(y,p)-reported.R2)<.00001
print('PASS holdout dates, unique postcode-month keys, nonnegative predictions, and metrics')
clusters=pd.read_csv(res/'postcode_residual_clusters.csv',dtype={'postcode':str})
summary=pd.read_csv(res/'residual_cluster_summary.csv')
assert len(clusters)==2810 and clusters.postcode.nunique()==2810
assert clusters.cluster.nunique()==meta['cluster_k']
assert summary.postcodes.sum()==2810
assert list(range(2016,2025))==[int(c) for c in clusters.columns if c.isdigit()]
assert len(pd.read_csv(res/'residual_cluster_k_diagnostics.csv'))==4
for f in ['M5_Figure_3_count_actual_vs_predicted.png', 'M5_Figure_4_count_model_comparison.png',
          'M5_Figure_5_count_annual_error.png','M5_Figure_6_count_residual_clusters.png',
          'M5_Figure_6b_residual_PCA.png']:
    assert (res/f).stat().st_size>15000
print('PASS residual clustering coverage, K diagnostics, and five rendered figures')
assert len(pd.read_csv(res/'ridge_search.csv'))==3
assert len(pd.read_csv(res/'random_forest_search.csv'))==4
print('PASS 3 ridge and 4 random-forest tuning candidates saved')
print('ALL CHECKS PASSED')
