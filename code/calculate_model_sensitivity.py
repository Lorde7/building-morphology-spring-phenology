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
for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, shutil
import numpy as np, pandas as pd, numba, statsmodels.api as sm
from scipy.stats import pearsonr
from model_paths import OUT
from prepare_vegetation_model_inputs import MODELS as TRENDY
from calculate_vegetation_model_sensitivity import regress, optimized, metric
RESULT = OUT / 'ST_comparison_all_models'
YEARS = np.arange(2001, 2021)

def stack(a):
    return np.array([[a[year - 2000, month - 1] for (year, month) in [(yr - 1, 10), (yr - 1, 11), (yr - 1, 12)] + [(yr, m) for m in range(1, 7)]] for yr in YEARS], dtype=np.float32)[:, :, None, :]

def common_predictors(lag, mean, t, p, r):
    import run_preseason_pipeline_wind_grid as base
    result = []
    for (cube, total) in [(t, False), (p, True), (r, True)]:
        a = base.extract_preseason_sos(np.asarray(lag)[None, :], np.asarray(mean)[None, :], cube, total)[:, 0, :]
        a[a == base.OUT_NODATA] = np.nan
        result.append(a)
    return result

def performance(frame, model, family):
    row = metric(frame, model, 'strict15_shared_satellite_OPL')
    row['family'] = family
    return row

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    numba.set_num_threads(4)
    RESULT.mkdir(exist_ok=True)
    (RESULT / 'code').mkdir(exist_ok=True)
    grid = pd.read_csv(OUT / 'common_3degree_cells_containing_cities.csv', dtype={'cell_id': str})
    keys = grid.cell_id.tolist()
    n = len(keys)
    fields = {v: np.full((21, 12, n), np.nan) for v in ['tmin', 'tmax', 'ppt', 'srad']}
    for variable in fields:
        for year in range(2000, 2021):
            q = pd.read_csv(OUT / 'monthly_observed_climate_area3degree' / f'{variable}_{year}.csv', dtype={'cell_id': str})
            fields[variable][year - 2000] = q.pivot(index='month', columns='cell_id', values='value').reindex(index=range(1, 13), columns=keys)
    tc = stack((fields['tmin'] + fields['tmax']) / 2)
    pc = stack(fields['ppt'])
    rc = stack(fields['srad'])
    observed = pd.concat([pd.read_csv(OUT / 'satellite_SOS' / f'SOS_{year}.csv', dtype={'cell_id': str}) for year in YEARS])
    yobs = observed.pivot(index='year', columns='cell_id', values='SOS').reindex(index=YEARS, columns=keys).to_numpy(float)
    (satlag, satscore, satmean, satpredictors) = optimized(yobs, tc, pc, rc)
    sat = pd.DataFrame([dict(cell_id=cell, OPL=satlag[i], OPL_score=satscore[i], SOS_mean=satmean[i], **regress(yobs[:, i], *[x[:, i] for x in satpredictors])) for (i, cell) in enumerate(keys)])
    sat.to_csv(RESULT / 'satellite_3degree_ST.csv', index=False, encoding='utf-8-sig')
    satraw = observed.copy()
    satraw.to_csv(RESULT / 'satellite_2001_2020_SOS.csv', index=False, encoding='utf-8-sig')
    from model_parameters import CMIP6_MODELS
    cmip = list(CMIP6_MODELS)
    models = [('TRENDY', m) for m in TRENDY] + [('CMIP6', m) for m in cmip]
    pairs = []
    own = []
    scores = []
    coverage = []
    checks = []
    annual = []
    model_inputs = {}
    for (family, model) in models:
        folder = OUT if family == 'CMIP6' else OUT / 'TRENDY_common_3degree'
        dates = pd.read_csv(folder / 'model_SOS' / (model + '.csv'), dtype={'cell_id': str})
        dates['family'] = family
        dates['model'] = model
        annual.append(dates)
        yy = dates.pivot(index='year', columns='cell_id', values='SOS_15').reindex(index=YEARS, columns=keys).to_numpy(float)
        yy[(yy < 1) | (yy >= 190)] = np.nan
        if family == 'CMIP6':
            cubes = {v: np.full((21, 12, n), np.nan) for v in ['tas', 'pr', 'rsds']}
            for v in cubes:
                q = pd.read_csv(OUT / 'monthly_model_fields' / model / (v + '.csv'), dtype={'cell_id': str})
                for year in range(2000, 2021):
                    cubes[v][year - 2000] = q[q.year.eq(year)].pivot(index='month', columns='cell_id', values='value').reindex(index=range(1, 13), columns=keys)
            (t, p, r) = (stack(cubes['tas']), stack(cubes['pr']), stack(cubes['rsds']))
        else:
            (t, p, r) = (tc, pc, rc)
        shared = common_predictors(satlag, satmean, t, p, r)
        (ownlag, ownscore, ownmean, ownpred) = optimized(yy, t, p, r)
        model_inputs[family, model] = (yy, shared, ownlag, ownmean)
        permodel = []
        for (i, cell) in enumerate(keys):
            valid = np.isfinite(yy[:, i]) & np.isfinite(yobs[:, i])
            valid &= np.all(np.isfinite(np.column_stack([a[:, i] for a in shared + satpredictors])), axis=1)
            a = regress(np.where(valid, yobs[:, i], np.nan), *[x[:, i] for x in satpredictors])
            b = regress(np.where(valid, yy[:, i], np.nan), *[x[:, i] for x in shared])
            permodel.append(dict(family=family, model=model, method='strict15_shared_satellite_OPL', cell_id=cell, satellite_OPL=satlag[i], model_own_OPL=ownlag[i], SOS_observed_mean=float(satmean[i]), SOS_model_mean=float(ownmean[i]), ST_observed_matched=a['ST'], ST_model_matched=b['ST'], ST_error=b['ST'] - a['ST'], P_observed=a['Pval_T'], P_model=b['Pval_T'], SE_observed=a['SE_T'], SE_model=b['SE_T'], n_joint_years=b['n_years']))
            own.append(dict(family=family, model=model, cell_id=cell, model_own_OPL=ownlag[i], OPL_score=ownscore[i], SOS_mean=ownmean[i], **regress(yy[:, i], *[x[:, i] for x in ownpred])))
            if i in [0, n // 4, n // 2, 3 * n // 4, n - 1] and b['n_years'] >= 13 and np.isfinite(b['ST']):
                X = sm.add_constant(np.column_stack([x[:, i] for x in shared])[valid])
                if np.linalg.matrix_rank(X) == 4:
                    fit = sm.OLS(yy[valid, i], X).fit()
                    error = abs(float(fit.params[1]) - b['ST'])
                    assert error < 1e-09, (model, cell, error)
                    checks.append(dict(family=family, model=model, cell_id=cell, reference_ST=float(fit.params[1]), pipeline_ST=b['ST'], absolute_error=error, n_years=b['n_years']))
        q = pd.DataFrame(permodel)
        pairs.append(q)
        scores.append(performance(q, model, family))
        coverage.append(dict(family=family, model=model, n_input_cells=n, n_valid_spring_dates=int(np.isfinite(yy).sum()), n_13plus_year_cells=int((np.isfinite(yy).sum(axis=0) >= 13).sum()), n_paired_ST_cells=int(q.ST_model_matched.notna().sum()), ST_P_selection=False))
        print('MODEL_ST', model, 'n', scores[-1]['n_common_cells'], 'RMSE', round(scores[-1]['RMSE_ST'], 3), 'r', round(scores[-1]['Pearson_r'], 3), flush=True)
    pair = pd.concat(pairs, ignore_index=True)
    allown = pd.DataFrame(own)
    pair.to_csv(RESULT / 'all_models_shared_window_paired_ST.csv', index=False, encoding='utf-8-sig')
    allown.to_csv(RESULT / 'all_models_own_optimized_OPL_ST.csv', index=False, encoding='utf-8-sig')
    pd.concat(annual, ignore_index=True).to_csv(RESULT / 'all_models_annual_SOS.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(RESULT / 'model_performance_available_cells.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(RESULT / 'model_phenology_ST_coverage.csv', index=False, encoding='utf-8-sig')
    common_year = np.isfinite(yobs) & np.all(np.isfinite(np.stack(satpredictors)), axis=0)
    for (yy, shared, _, _) in model_inputs.values():
        common_year &= np.isfinite(yy) & np.all(np.isfinite(np.stack(shared)), axis=0)
    common_pairs = []
    for ((family, model), (yy, shared, ownlag, ownmean)) in model_inputs.items():
        for (i, cell) in enumerate(keys):
            valid = common_year[:, i]
            a = regress(np.where(valid, yobs[:, i], np.nan), *[x[:, i] for x in satpredictors])
            b = regress(np.where(valid, yy[:, i], np.nan), *[x[:, i] for x in shared])
            common_pairs.append(dict(family=family, model=model, method='strict15_all_models_same_years_shared_OPL', cell_id=cell, satellite_OPL=satlag[i], model_own_OPL=ownlag[i], SOS_observed_mean=float(satmean[i]), SOS_model_mean=float(ownmean[i]), ST_observed_matched=a['ST'], ST_model_matched=b['ST'], ST_error=b['ST'] - a['ST'], P_observed=a['Pval_T'], P_model=b['Pval_T'], SE_observed=a['SE_T'], SE_model=b['SE_T'], n_joint_years=b['n_years']))
    comparable = pd.DataFrame(common_pairs)
    comparable.to_csv(RESULT / 'all_models_common_years_shared_window_ST.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([dict(cell_id=cell, year=int(year), valid_all20=bool(common_year[j, i])) for (i, cell) in enumerate(keys) for (j, year) in enumerate(YEARS)]).to_csv(RESULT / 'all_models_common_valid_year_mask.csv', index=False, encoding='utf-8-sig')
    common = set(keys)
    for (_, q) in comparable.groupby('model'):
        common &= set(q.loc[q.ST_model_matched.notna() & q.ST_observed_matched.notna(), 'cell_id'])
    same = comparable[comparable.cell_id.isin(common)].copy()
    assert same.groupby('cell_id').ST_observed_matched.apply(lambda x: np.ptp(x.to_numpy())).max() < 1e-10
    same.to_csv(RESULT / 'all_models_same_complete_cells_paired_ST.csv', index=False, encoding='utf-8-sig')
    common_scores = [performance(q, m, q.family.iloc[0]) for (m, q) in same.groupby('model')]
    pd.DataFrame(common_scores).to_csv(RESULT / 'all_models_same_cells_performance.csv', index=False, encoding='utf-8-sig')
    ensembles = []
    ensemble_metrics = []
    for (label, group) in [('TRENDY', same[same.family.eq('TRENDY')]), ('CMIP6', same[same.family.eq('CMIP6')]), ('AllModels', same)]:
        q = group.groupby('cell_id').agg(ST_observed_matched=('ST_observed_matched', 'mean'), ST_model_matched=('ST_model_matched', 'mean'), n_models=('model', 'nunique'), SOS_observed_mean=('SOS_observed_mean', 'mean'), SOS_model_mean=('SOS_model_mean', 'mean')).reset_index()
        q['family'] = label
        ensembles.append(q)
        ensemble_metrics.append(performance(q, label, label))
    pd.concat(ensembles, ignore_index=True).to_csv(RESULT / 'ensemble_ST_same_cells.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(ensemble_metrics).to_csv(RESULT / 'ensemble_performance_same_cells.csv', index=False, encoding='utf-8-sig')
    summary = dict(n_cities=638, n_independent_3degree_cells=n, n_models=len(models), n_all_models_common_cells=len(common))
    summary.update(ensemble_metrics=ensemble_metrics)
    (RESULT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ALL_MODELS_COMPLETE', json.dumps(summary, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
