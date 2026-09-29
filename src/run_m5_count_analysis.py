#!/usr/bin/env python3
"""Module 5 conditional count-only analysis, NOT an adoption-rate analysis.

Task: fit using 2008-2015 monthly postcode counts and estimate month-t counts
using ACTUAL lagged observed counts through t-1 even after 2015. This is a
rolling-information one-month-ahead backtest with frozen model weights, not a
fixed-origin 2015 multi-year forecast. No ABS demographics/dwelling denominators
or postcode boundaries were supplied. Outputs must not be presented as household
adoption rates or completed geographic analysis.
"""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import make_scorer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.build_panel import load_panel

SEED=42
FEATURES=['lag_1','lag_2','lag_3','lag_6','lag_12','prior_3mo_mean',
          'prior_12mo_mean','prior_cumulative','month_sin','month_cos']


def load_features(path):
    # Use the repository loader so postcode dtype and the metadata sidecar are
    # validated consistently with the shared data pipeline.
    df=load_panel(path)
    expected={'Small Unit Installation Postcode','month','installations'}
    if set(df.columns)!=expected: raise ValueError('Unexpected source columns')
    df=df.rename(columns={'Small Unit Installation Postcode':'postcode'})
    if df[['postcode','month','installations']].isna().any().any(): raise ValueError('Missing input')
    if not df.postcode.str.fullmatch(r'\d{4}').all(): raise ValueError('Postcode not four digits')
    if df.duplicated(['postcode','month']).any(): raise ValueError('Duplicate postcode-month')
    if (df.installations<0).any() or (df.installations%1!=0).any(): raise ValueError('Invalid count')
    excluded=df.loc[df.postcode.eq('0000'),'installations'].sum()
    if excluded != 4: raise ValueError('0000 source discrepancy; recheck before running')
    df=df.loc[df.postcode.ne('0000')].sort_values(['postcode','month']).reset_index(drop=True)
    g=df.groupby('postcode',sort=False).installations
    old=g.shift(1)
    for n in [1,2,3,6,12]: df[f'lag_{n}']=g.shift(n).astype('float32')
    pg=old.groupby(df.postcode,sort=False)
    for n in [3,12]:
        df[f'prior_{n}mo_mean']=pg.transform(lambda v:v.rolling(n,min_periods=n).mean()).astype('float32')
    df['prior_cumulative']=(g.cumsum()-df.installations).astype('float32')
    mm=df.month.dt.month.to_numpy()
    df['month_sin']=np.sin(2*np.pi*mm/12).astype('float32')
    df['month_cos']=np.cos(2*np.pi*mm/12).astype('float32')
    df=df.loc[df.month.between('2008-01-01','2024-12-01')].dropna(subset=FEATURES).copy()
    if df.month.min()!=pd.Timestamp('2008-01-01') or df.month.max()!=pd.Timestamp('2024-12-01'):
        raise ValueError('Unexpected analysis period')
    return df,int(excluded)


def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),0,None)
    return {'MAE':float(mean_absolute_error(y,p)),
            'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)), 'actual_total':int(np.sum(y)),
            'predicted_total':float(p.sum())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=ROOT/'data/processed/cer_solar_panel_long.csv')
    ap.add_argument('--out',type=Path,default=ROOT/'results/experiment1')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); df,excluded=load_features(args.input)
    train=df.loc[df.month.lt('2016-01-01')].copy()
    test=df.loc[df.month.ge('2016-01-01')].copy()
    assert train.month.max()==pd.Timestamp('2015-12-01')
    assert test.month.min()==pd.Timestamp('2016-01-01')
    assert len(train)==269664 and len(test)==303480
    X=train[FEATURES].to_numpy(dtype='float32'); y=train.installations.to_numpy(dtype='float32')
    XT=test[FEATURES].to_numpy(dtype='float32'); yt=test.installations.to_numpy(dtype='float32')
    dates=train.month.to_numpy()
    # Two expanding, whole-year splits. Never split randomly across postcodes/months.
    folds=[(np.flatnonzero(dates<np.datetime64('2014-01-01')),
            np.flatnonzero((dates>=np.datetime64('2014-01-01'))&(dates<np.datetime64('2015-01-01')))),
           (np.flatnonzero(dates<np.datetime64('2015-01-01')),
            np.flatnonzero((dates>=np.datetime64('2015-01-01'))&(dates<np.datetime64('2016-01-01'))))]
    assert all(dates[a].max()<dates[b].min() for a,b in folds)
    # MAE search: scorer uses raw signed predictions; held-out metrics explicitly clip
    # negatives to zero. This discrepancy is documented and checked in metadata.
    ridge=Pipeline([('scaler',StandardScaler()),('model',Ridge())])
    rf=RandomForestRegressor(random_state=SEED,n_jobs=2,max_samples=.65,max_features=.9)
    searches={
      'Ridge':GridSearchCV(ridge,{'model__alpha':[.1,20.,200.]},scoring='neg_mean_absolute_error',
                           cv=folds,refit=False,n_jobs=1,error_score='raise',return_train_score=False),
      'Random forest':GridSearchCV(rf,{'n_estimators':[25], 'max_depth':[10,14],
                              'min_samples_leaf':[10,20]},scoring='neg_mean_absolute_error',
                              cv=folds,refit=False,n_jobs=1,error_score='raise',return_train_score=False)
    }
    for name, search in searches.items():
        t=time.monotonic(); print('TUNING',name,'begin',flush=True)
        search.fit(X,y)
        table=pd.DataFrame(search.cv_results_)
        table['model']=name
        wanted=['model','params','mean_test_score','std_test_score','rank_test_score',
                'split0_test_score','split1_test_score','mean_fit_time']
        table[wanted].to_csv(args.out/(name.replace(' ','_').lower()+'_search.csv'),index=False)
        print('TUNED',name,'BEST',search.best_params_,'CV_MAE_RAW',-search.best_score_,
              'SECONDS',round(time.monotonic()-t,2),flush=True)
    trained={}
    for name,search in searches.items():
        est=clone(search.estimator).set_params(**search.best_params_)
        print('FIT_START',name,flush=True)
        t=time.monotonic(); est.fit(X,y)
        print('FIT_END_PREDICT_START',name,flush=True)
        trained[name]=np.clip(est.predict(XT),0,None)
        print('FIT',name,'SECONDS',round(time.monotonic()-t,2),flush=True)
    preds={'Previous month':test.lag_1.to_numpy(),
           'Same month last year':test.lag_12.to_numpy(),**trained}
    print('PREDICTIONS_DONE',flush=True)
    scores=[]
    for name,p in preds.items():scores.append(dict(model=name,observations=len(yt),**metrics(yt,p)))
    pd.DataFrame(scores).to_csv(args.out/'holdout_model_metrics.csv',index=False,float_format='%.6f')
    forecast=test[['postcode','month','installations']].rename(columns={'installations':'actual'}).reset_index(drop=True)
    for name,p in preds.items():forecast[name]=p
    forecast.to_csv(args.out/'holdout_predictions.csv.gz',index=False,compression='gzip',float_format='%.6f')
    annual=[]
    for year,frame in forecast.groupby(forecast.month.dt.year):
        for name in preds:
            annual.append(dict(year=int(year),model=name,observations=len(frame),
                               **metrics(frame.actual.to_numpy(),frame[name].to_numpy())))
    pd.DataFrame(annual).to_csv(args.out/'holdout_metrics_by_year.csv',index=False,float_format='%.6f')
    fig,ax=plt.subplots(figsize=(10,5.3))
    monthly=forecast.groupby('month')[['actual','Ridge','Random forest','Previous month']].sum()
    monthly.to_csv(args.out/'holdout_national_monthly_totals.csv')
    for label in monthly.columns:ax.plot(monthly.index,monthly[label],label=label,linewidth=1.2)
    ax.set(title='Preliminary count-only: observed vs. one-month-ahead model totals (2016–2024)',
           xlabel='Month',ylabel='Summed monthly installations (not adoption rate)')
    ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(args.out/'M5_Figure_3_count_actual_vs_predicted.png',dpi=175);plt.close(fig)
    mm=pd.DataFrame(scores)
    fig,ax=plt.subplots(figsize=(9,5))
    xx=np.arange(len(mm));w=.37
    ax.bar(xx-w/2,mm.MAE,w,label='MAE');ax.bar(xx+w/2,mm.RMSE,w,label='RMSE')
    ax.set_xticks(xx,mm.model,rotation=13,ha='right')
    ax.set(title='Preliminary count-only: held-out prediction errors',ylabel='Installations per postcode-month')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(args.out/'M5_Figure_4_count_model_comparison.png',dpi=175);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    yearly=pd.DataFrame(annual)
    for name,seg in yearly.groupby('model',sort=False):ax.plot(seg.year,seg.MAE,label=name,marker='o',linewidth=1.5)
    ax.set(title='Preliminary count-only: held-out MAE by year',xlabel='Year',ylabel='MAE: installations/postcode-month')
    ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(args.out/'M5_Figure_5_count_annual_error.png',dpi=175);plt.close(fig)
    print('START_CLUSTERING',flush=True)
    # Cluster annual mean SIGNED residuals of the tuned forest. Keeps magnitude
    # and direction of errors; PCA/standardization is across postcode rows, not
    # independently within each postcode (which would erase sign and magnitude).
    forecast['residual']=forecast['actual']-forecast['Random forest']
    forecast['year']=forecast.month.dt.year
    profiles=forecast.groupby(['postcode','year']).residual.mean().unstack('year')
    assert profiles.shape==(2810,9) and not profiles.isna().any().any()
    standardized=StandardScaler().fit_transform(profiles.to_numpy())
    pca=PCA(n_components=3,random_state=SEED).fit(standardized)
    embedded=pca.transform(standardized)
    evaluations=[];clusterers={}
    for k in [2,3,4,5]:
        print('TRY_CLUSTER',k,flush=True)
        km=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit(embedded)
        score=silhouette_score(embedded,km.labels_,sample_size=1000,random_state=SEED)
        evaluations.append({'k':k,'silhouette_sample1000':score,'inertia':km.inertia_})
        clusterers[k]=km
    evaluation=pd.DataFrame(evaluations);evaluation.to_csv(args.out/'residual_cluster_k_diagnostics.csv',index=False)
    chosen=int(evaluation.loc[evaluation.silhouette_sample1000.idxmax(),'k'])
    labels=clusterers[chosen].labels_
    clusters=profiles.copy();clusters['cluster']=labels
    clusters['mean_signed_residual']=profiles.mean(axis=1)
    clusters['mean_absolute_annual_residual']=profiles.abs().mean(axis=1)
    clusters=clusters.reset_index()
    clusters.to_csv(args.out/'postcode_residual_clusters.csv',index=False,float_format='%.6f')
    summary=clusters.groupby('cluster').agg(postcodes=('postcode','count'),
        mean_signed_residual=('mean_signed_residual','mean'),
        mean_absolute_annual_residual=('mean_absolute_annual_residual','mean')).reset_index()
    summary.to_csv(args.out/'residual_cluster_summary.csv',index=False,float_format='%.6f')
    profiles_w=clusters.melt(id_vars=['postcode','cluster','mean_signed_residual','mean_absolute_annual_residual'],
                              value_vars=list(range(2016,2025)),var_name='year',value_name='signed_error')
    fig,ax=plt.subplots(figsize=(10,5.5))
    for number,seg in profiles_w.groupby('cluster',sort=True):
        v=seg.groupby('year').signed_error.mean()
        ax.plot(v.index.astype(int),v.values,marker='o',label=f'Cluster {number} (n={(clusters.cluster==number).sum()})')
    ax.axhline(0,linewidth=1,color='gray',linestyle='--')
    ax.set(title='Exploratory clusters of annual out-of-sample COUNT residuals',
           xlabel='Held-out year',ylabel='Mean observed − predicted monthly installations/postcode')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(args.out/'M5_Figure_6_count_residual_clusters.png',dpi=175);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for number in sorted(set(labels)):
        mask=labels==number
        ax.scatter(embedded[mask,0],embedded[mask,1],s=7,alpha=.43,label=f'Cluster {number}')
    ax.set(title='PCA of 2016–2024 annual count-error profiles',xlabel='PC1',ylabel='PC2')
    ax.legend(markerscale=2,fontsize=8);fig.tight_layout()
    fig.savefig(args.out/'M5_Figure_6b_residual_PCA.png',dpi=175);plt.close(fig)
    report={'data':'CER monthly solar-installation counts; no ABS, no adoption denominator',
      'count_only':True,'train':'2008-01 to 2015-12','holdout':'2016-01 to 2024-12',
      'rolling_one_month_ahead_uses_observed_post_2015_lags':True,
      'geographic_map_completed':False,'residential_filter_completed':False,
      'privacy_floor_observed_cell_flags_available':False,
      'bayessearchcv_run':False,'bayessearchcv_reason':'scikit-optimize not installed; full Bayesian search not run',
      'search_criterion':'raw-prediction MAE (not clipped); fixed 2 expanding full-year folds: 2014 and 2015',
      'heldout_metric_predictions_clipped_to_nonnegative':True,
      'optimal_params':{name:search.best_params_ for name,search in searches.items()},
      'search_cv_mae':{name:float(-search.best_score_) for name,search in searches.items()},
      'rows_train':len(train),'rows_holdout':len(test),'postcodes':forecast.postcode.nunique(),
      'excluded_0000_installations':excluded,'cluster_features':'2016–2024 annual mean signed residuals of tuned random forest count model',
      'cluster_k':chosen,'pca_variance_explained':pca.explained_variance_ratio_.tolist(),
      'elapsed_seconds':round(time.monotonic()-start,1),'sklearn':sklearn.__version__,
      'pandas':pd.__version__,'numpy':np.__version__,'seed':SEED}
    (args.out/'run_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print('OVERALL METRICS\n',pd.DataFrame(scores).round(4).to_string(index=False),flush=True)
    print('CLUSTERS\n',summary.round(4).to_string(index=False),flush=True)
    print('K DIAGNOSTICS\n',evaluation.round(4).to_string(index=False),flush=True)
    print('COMPLETE_SECS',report['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
