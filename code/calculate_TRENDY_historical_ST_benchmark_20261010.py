import os as _os, json as _json
from pathlib import Path as _Path

def configured_path(key):
    if not _os.environ.get('MORPH_INPUT_CONFIG'):
        raise RuntimeError('Configure MORPH_INPUT_CONFIG for archived experiment input/output paths: ' + key)
    config = _json.loads(_Path(_os.environ['MORPH_INPUT_CONFIG']).read_text(encoding='utf8'))
    if key not in config:
        raise RuntimeError('Missing configured path: ' + key)
    return config[key]
                                                                            
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[k] = '1'
import sys, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, t as student_t
import numba
import run_preseason_pipeline_wind_grid as base
from prepare_TRENDY_historical_ST_20261010 import OUT, ROOT, MODELS
YEARS = np.arange(2001, 2021)

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')

def regress(y, tt, pp, rr):
    valid = np.isfinite(y) & np.isfinite(tt) & np.isfinite(pp) & np.isfinite(rr)
    n = int(valid.sum())
    if n < 13:
        return dict(ST=np.nan, Pval_T=np.nan, SE_T=np.nan, n_years=n, R2=np.nan)
    X = np.column_stack([np.ones(n), tt[valid], pp[valid], rr[valid]])
    yy = y[valid]
    if np.ptp(yy) < 1e-10:
        return dict(ST=0.0, Pval_T=np.nan, SE_T=0.0, n_years=n, R2=np.nan)
    scales = X[:, 1:].std(axis=0)
    scales[scales == 0] = 1
    zz = (X[:, 1:] - X[:, 1:].mean(axis=0)) / scales
    xx = np.column_stack([np.ones(n), zz])
    if np.linalg.matrix_rank(xx) < 4:
        return dict(ST=np.nan, Pval_T=np.nan, SE_T=np.nan, n_years=n, R2=np.nan)
    beta = np.linalg.lstsq(xx, yy, rcond=None)[0]
    resid = yy - xx @ beta
    se = np.sqrt(np.diag(np.linalg.inv(xx.T @ xx)) * np.sum(resid ** 2) / (n - 4))
    native = beta[1] / scales[0]
    native_se = se[1] / scales[0]
    P = float(2 * student_t.sf(abs(native / native_se), n - 4)) if native_se > 0 else 0.0
    sst = np.sum((yy - yy.mean()) ** 2)
    return dict(ST=float(native), Pval_T=P, SE_T=float(native_se), n_years=n, R2=float(1 - np.sum(resid ** 2) / sst) if sst > 0 else np.nan)

def optimized(y, t, p, r):
    y3 = y[:, None, :].astype(np.float32)
    (lag, score) = base.compute_opl_sos(y3, t, p, r)
    mean = base.mean_doy(y3)
    predictors = [base.extract_preseason_sos(lag, mean, a, flag)[:, 0, :] for (a, flag) in [(t, False), (p, True), (r, True)]]
    for a in predictors:
        a[a == base.OUT_NODATA] = np.nan
    return (lag.ravel(), score.ravel(), mean.ravel(), predictors)

def metric(frame, model, method):
    q = frame.dropna(subset=['ST_observed_matched', 'ST_model_matched'])
    obs = q.ST_observed_matched.to_numpy()
    pred = q.ST_model_matched.to_numpy()
    error = pred - obs
    n = len(q)
    corr = float(pearsonr(obs, pred)[0]) if n > 2 and np.std(obs) > 0 and (np.std(pred) > 0) else np.nan
    return dict(model=model, method=method, n_common_cells=n, RMSE_ST=float(np.sqrt(np.mean(error ** 2))) if n else np.nan, MAE_ST=float(np.mean(abs(error))) if n else np.nan, mean_bias_ST=float(np.mean(error)) if n else np.nan, median_bias_ST=float(np.median(error)) if n else np.nan, Pearson_r=corr, predictive_R2=float(1 - np.sum(error ** 2) / np.sum((obs - obs.mean()) ** 2)) if n and np.std(obs) > 0 else np.nan, median_observed_ST=float(np.median(obs)) if n else np.nan, median_model_ST=float(np.median(pred)) if n else np.nan, same_sign_fraction=float(np.mean(np.sign(obs) == np.sign(pred))) if n else np.nan, negative_model_fraction=float(np.mean(pred < 0)) if n else np.nan, median_SOS_bias_days=float(np.nanmedian(q.SOS_model_mean - q.SOS_observed_mean)) if n else np.nan)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    numba.set_num_threads(4)
    grid = pd.read_csv(OUT / 'common_2degree_cells_containing_cities.csv', dtype={'cell_id': str})
    keys = grid.cell_id.tolist()
    n = len(grid)
    cubes = {v: np.full((21, 12, n), np.nan) for v in ['tmin', 'tmax', 'ppt', 'srad']}
    for v in cubes:
        for year in range(2000, 2021):
            q = pd.read_csv(OUT / 'monthly_climate_area2degree' / f'{v}_{year}.csv', dtype={'cell_id': str})
            cubes[v][year - 2000] = q.pivot(index='month', columns='cell_id', values='value').reindex(index=range(1, 13), columns=keys).to_numpy()
    temp = (cubes['tmin'] + cubes['tmax']) / 2

    def stack(a):
        return np.array([[a[year - 2000, month - 1] for (year, month) in [(yr - 1, 10), (yr - 1, 11), (yr - 1, 12)] + [(yr, m) for m in range(1, 7)]] for yr in YEARS], dtype=np.float32)[:, :, None, :]
    (t, p, r) = (stack(temp), stack(cubes['ppt']), stack(cubes['srad']))
    observed = pd.concat([pd.read_csv(OUT / 'satellite_SOS' / f'SOS_{yr}.csv', dtype={'cell_id': str}) for yr in YEARS])
    yobs = observed.pivot(index='year', columns='cell_id', values='SOS').reindex(index=YEARS, columns=keys).to_numpy(float)
    (satlag, satscore, satmean, spredictors) = optimized(yobs, t, p, r)
    sat = []
    for (i, cell) in enumerate(keys):
        sat.append(dict(cell_id=cell, OPL=satlag[i], OPL_score=satscore[i], SOS_mean=satmean[i], **regress(yobs[:, i], *[a[:, i] for a in spredictors])))
    pd.DataFrame(sat).to_csv(OUT / 'satellite_2degree_ST.csv', index=False, encoding='utf-8-sig')
    allpairs = []
    allown = []
    metrics = []
    coverage = []
    for model in MODELS:
        dates = pd.read_csv(OUT / 'model_SOS' / f'{model}.csv', dtype={'cell_id': str})
        for (method, column) in [('strict_15', 'SOS_15'), ('phenofit_TRS15', 'SOS_phenofit_TRS15'), ('diagnostic_max_slope', 'SOS_derivative'), ('diagnostic_max_acceleration', 'SOS_inflection')]:
            yy = dates.pivot(index='year', columns='cell_id', values=column).reindex(index=YEARS, columns=keys).to_numpy(float)
            yy[(yy < 1) | (yy >= 190)] = np.nan
            (lag, score, mean, predictors) = optimized(yy, t, p, r)
            pairs = []
            own = []
            for (i, cell) in enumerate(keys):
                joint = np.isfinite(yy[:, i]) & np.isfinite(yobs[:, i])
                rowobs = regress(np.where(joint, yobs[:, i], np.nan), *[a[:, i] for a in spredictors])
                rowmodel = regress(np.where(joint, yy[:, i], np.nan), *[a[:, i] for a in spredictors])
                rec = dict(model=model, method=method, cell_id=cell, satellite_OPL=satlag[i], model_own_OPL=lag[i], SOS_observed_mean=float(satmean[i]), SOS_model_mean=float(mean[i]), ST_observed_matched=rowobs['ST'], ST_model_matched=rowmodel['ST'], ST_error=rowmodel['ST'] - rowobs['ST'], P_observed=rowobs['Pval_T'], P_model=rowmodel['Pval_T'], n_joint_years=rowmodel['n_years'])
                pairs.append(rec)
                own.append(dict(model=model, method=method, cell_id=cell, OPL=lag[i], OPL_score=score[i], SOS_mean=mean[i], **regress(yy[:, i], *[a[:, i] for a in predictors])))
            q = pd.DataFrame(pairs)
            allpairs.append(q)
            allown.append(pd.DataFrame(own))
            metrics.append(metric(q, model, method))
            coverage.append(dict(model=model, method=method, raw_finite_dates=int(np.isfinite(dates[column]).sum()), valid_spring_dates=int(np.isfinite(yy).sum()), cells_with_13plus_dates=int((np.isfinite(yy).sum(axis=0) >= 13).sum()), common_complete_ST_cells=int(q.ST_model_matched.notna().sum())))
        print('MODEL_ST_COMPLETE', model, metrics[-4], flush=True)
    pair = pd.concat(allpairs, ignore_index=True)
    own = pd.concat(allown, ignore_index=True)
    scores = pd.DataFrame(metrics)
    pair.to_csv(OUT / 'all_models_joint_year_shared_OPL_ST_comparison.csv', index=False, encoding='utf-8-sig')
    own.to_csv(OUT / 'all_models_own_OPL_ST_sensitivity.csv', index=False, encoding='utf-8-sig')
    scores.to_csv(OUT / 'all_methods_model_ST_performance.csv', index=False, encoding='utf-8-sig')
    main = scores[scores.method.eq('strict_15')].copy()
    main.to_csv(OUT / 'main_15_models_ST_performance.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(OUT / 'model_date_ST_coverage.csv', index=False, encoding='utf-8-sig')
    strict = pair[pair.method.eq('strict_15')]
    common = set(keys)
    for (_, g) in strict.groupby('model'):
        common &= set(g.loc[g.ST_model_matched.notna() & g.ST_observed_matched.notna(), 'cell_id'])
    same = [metric(g[g.cell_id.isin(common)], m, 'strict_15_all_models_same_cells') for (m, g) in strict.groupby('model')]
    pd.DataFrame(same).to_csv(OUT / 'all_15_models_common_cells_performance.csv', index=False, encoding='utf-8-sig')
    summary = dict(n_input_cities=638, n_coarse_cells=n, n_satellite_ST_cells=int(pd.DataFrame(sat).ST.notna().sum()), n_all_models_common_ST_cells=len(common), models=main.to_dict('records'))
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
