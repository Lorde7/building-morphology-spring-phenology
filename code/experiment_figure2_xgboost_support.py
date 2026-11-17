from pathlib import Path
import os, json
PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT = PACKAGE / 'outputs' / 'experiment_figure2'
CONFIG = json.loads(Path(os.environ['MORPH_INPUT_CONFIG']).read_text(encoding='utf8')) if os.environ.get('MORPH_INPUT_CONFIG') else {}

def input_path(key, relative):
    return Path(CONFIG[key]) if key in CONFIG else PACKAGE / 'data' / relative
                                                                               
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import sys, json, time, hashlib, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
WORK = PACKAGE
OUT = OUTPUT
EX = input_path('figure2_exclusion_root', 'experiment_figure2/exclusion636')
ENV = input_path('figure2_environment_root', 'experiment_figure2/environment')
META = input_path('figure2_city_metadata', 'experiment_figure2/city_metadata.csv')
THRESHOLDS = [10, 20, 30, 50, 100]
NWORK = max(1, (os.cpu_count() or 4) - 2)
TARGET = 'ST_OLS_days_per_C'
COUNT = 'ST_OLS_valid_pixel_count'

def jsave(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def metric(y, p):
    return dict(n=len(y), R2=float(r2_score(y, p)), RMSE=float(mean_squared_error(y, p) ** 0.5), MAE=float(mean_absolute_error(y, p)))

def affine(y, p):
    return np.linalg.lstsq(np.c_[np.ones(len(p)), p], y, rcond=None)[0]

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    start = time.perf_counter()
    for p in [OUT, OUT / 'code', OUT / 'QA', OUT / 'counts']:
        p.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, dtype={'city_id': str})
    d = pd.read_csv(ENV / 'raw_model_input_636_environment.csv', dtype={'city_id': str})
    orig = pd.read_csv(EX / 'grid_frozen/out_of_fold_predictions_636.csv', dtype={'city_id': str})
    assert len(meta) == 638 and len(d) == 636 and meta.city_id.is_unique and d.city_id.is_unique
    assert d.city_id.tolist() == orig.city_id.tolist()
    assert not d.city_id.isin(['35935', '39895']).any()
    cols = ['city_id', COUNT, TARGET, 'city_area_km2', 'lon_gpkg', 'lat_gpkg']
    counts = meta[cols].sort_values(COUNT).copy()
    counts['previous_trial_excluded'] = counts.city_id.isin(['35935', '39895'])
    for threshold in THRESHOLDS:
        counts[f'passes_n_ge_{threshold}'] = counts[COUNT] >= threshold
    counts.to_csv(OUT / 'counts/all638_city_ST_pixel_counts.csv', index=False, encoding='utf-8-sig')
    counts[~counts.previous_trial_excluded].to_csv(OUT / 'counts/remaining636_city_ST_pixel_counts.csv', index=False, encoding='utf-8-sig')
    bins = [0, 9, 19, 29, 49, 99, 499, 999, np.inf]
    dist = []
    for (label, frame) in [('original638', counts), ('previous_exclusion636', counts[~counts.previous_trial_excluded])]:
        n = pd.cut(frame[COUNT], bins=bins).value_counts(sort=False)
        dist.extend((dict(cohort=label, interval=str(k), n_cities=int(v)) for (k, v) in n.items()))
    pd.DataFrame(dist).to_csv(OUT / 'counts/pixel_count_distribution.csv', index=False, encoding='utf-8-sig')
    support = meta.set_index('city_id').loc[d.city_id, COUNT].to_numpy(int)
    assert np.all(support > 0)
    y = d[TARGET].to_numpy(float)
    outer = orig.outer_fold.to_numpy(int)
    from figure2_parameters import FEATURE_GROUPS, CANDIDATE_PARAMETERS
    groups = {k: FEATURE_GROUPS[k] for k in ['core18', 'climate24']}
    full = groups['climate24']
    matrices = {k: d[fs].to_numpy(np.float32) for (k, fs) in groups.items()}
    assert np.isfinite(y).all() and all((np.isfinite(a).all() for a in matrices.values()))
    ctx = {}
    composition = []
    restrict = []
    jobs = []
    for threshold in [0] + THRESHOLDS:
        ix = np.flatnonzero(support >= threshold)
        c = meta.set_index('city_id').loc[d.city_id.iloc[ix]]
        (rho, p) = spearmanr(d.BMSI_A.to_numpy()[ix], y[ix])
        composition.append(dict(minimum_valid_pixels=threshold, n_cities=len(ix), removed_from636=636 - len(ix), total_ST_pixels=int(support[ix].sum()), median_ST_pixels=float(np.median(support[ix])), positive_ST_cities=int((y[ix] > 0).sum()), mean_ST=float(y[ix].mean()), sd_ST=float(y[ix].std(ddof=1)), median_city_area_km2=float(c.city_area_km2.median()), median_latitude=float(c.lat_gpkg.median()), median_longitude=float(c.lon_gpkg.median()), BMSI_ST_spearman=float(rho), spearman_P=float(p)))
        removed = counts[counts.city_id.isin(d.city_id) & (counts[COUNT] < threshold)]
        removed.to_csv(OUT / 'counts' / f'excluded_from636_n_lt_{threshold}.csv', index=False, encoding='utf-8-sig')
        for name in groups:
            base = pd.read_csv(ENV / name / 'out_of_fold_predictions_636.csv', dtype={'city_id': str})
            assert base.city_id.tolist() == d.city_id.tolist()
            restrict.append(dict(minimum_valid_pixels=threshold, group=name, analysis='Restrict existing OOF evaluation only; no refit', **metric(y[ix], base.predicted_ST.to_numpy()[ix])))
        if threshold == 0:
            continue
        for fold in range(1, 6):
            train = ix[outer[ix] != fold]
            test = ix[outer[ix] == fold]
            inner = list(KFold(3, shuffle=True, random_state=20261014 + fold).split(train))
            innerid = np.zeros(len(train), int)
            for (k, (a, b)) in enumerate(inner, 1):
                innerid[b] = k
            ctx[threshold, fold] = dict(train=train, test=test, inner=inner, innerid=innerid)
            for name in groups:
                for (trial, pars) in enumerate(CANDIDATE_PARAMETERS[str(fold)], 1):
                    jobs.append((threshold, fold, name, trial, pars))
    pd.DataFrame(composition).to_csv(OUT / 'cohort_composition.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(restrict).to_csv(OUT / 'evaluation_restriction_without_refit.csv', index=False, encoding='utf-8-sig')
    print('SUPPORT_SCREEN_START', len(jobs), 'configurations', 3 * len(jobs), 'inner_fits', 'workers', NWORK, flush=True)
    records = []

    def score(job):
        (t, f, name, tr, pars) = job
        c = ctx[t, f]
        train = c['train']
        xx = matrices[name]
        pred = np.full(len(train), np.nan)
        rounds = []
        for (a, b) in c['inner']:
            p = dict(pars, n_estimators=1600, base_score=float(y[train[a]].mean()), early_stopping_rounds=80, eval_metric='rmse', n_jobs=1)
            if p['objective'] == 'reg:pseudohubererror':
                p['huber_slope'] = 2.0
            model = xgb.XGBRegressor(**p)
            model.fit(xx[train[a]], y[train[a]], eval_set=[(xx[train[b]], y[train[b]])], verbose=False)
            n = int(model.best_iteration + 1)
            rounds.append(n)
            pred[b] = model.get_booster().predict(xgb.DMatrix(xx[train[b]]), iteration_range=(0, n))
        assert np.isfinite(pred).all()
        return dict(threshold=t, fold=f, group=name, trial=tr, parameters=pars, refit_trees=int(np.median(rounds)), inner_rounds=rounds, inner=metric(y[train], pred), prediction=pred)
    with ThreadPoolExecutor(max_workers=NWORK) as pool:
        for (n, ft) in enumerate(as_completed([pool.submit(score, j) for j in jobs]), 1):
            records.append(ft.result())
            if n % 40 == 0 or n == len(jobs):
                print('SCREEN_PROGRESS', n, '/', len(jobs), 'seconds', round(time.perf_counter() - start, 1), flush=True)
    choices = []
    for ((t, f), c) in ctx.items():
        pergroup = {}
        for name in groups:
            top = sorted([r for r in records if (r['threshold'], r['fold'], r['group']) == (t, f, name)], key=lambda r: (r['inner']['RMSE'], r['trial']))[:3]
            single = top[0]['prediction']
            mean = np.mean([r['prediction'] for r in top], axis=0)
            cal = np.empty_like(mean)
            for k in range(1, 4):
                fit = c['innerid'] != k
                test = ~fit
                (a, b) = affine(y[c['train']][fit], mean[fit])
                cal[test] = a + b * mean[test]
            scores = {n: metric(y[c['train']], p) for (n, p) in [('best_single', single), ('top3_mean', mean), ('top3_calibrated', cal)]}
            variant = min(scores, key=lambda n: (scores[n]['RMSE'], n))
            (a, b) = affine(y[c['train']], mean)
            pergroup[name] = dict(variant=variant, inner_scores=scores, calibration=dict(intercept=float(a), slope=float(b)), trials=[dict(trial=r['trial'], parameters=dict(r['parameters'], n_estimators=r['refit_trees'])) for r in top])
            path = OUT / f'n_ge_{t}' / f'fold_{f:02d}'
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path / f'{name}_inner_OOF.npz', city_id=d.city_id.iloc[c['train']].to_numpy(str), inner_fold=c['innerid'], observed_ST=y[c['train']], best_single=single, top3_mean=mean, top3_calibrated=cal)
        selected = min(pergroup, key=lambda n: (pergroup[n]['inner_scores'][pergroup[n]['variant']]['RMSE'], n))
        choices.append(dict(threshold=t, fold=f, groups=pergroup, training_only_selected_group=selected))
    print('SCREEN_CHOICES_FROZEN', flush=True)
    refitjobs = [(ch['threshold'], ch['fold'], name, tr) for ch in choices for (name, info) in ch['groups'].items() for tr in info['trials']]

    def refit(job):
        (t, f, name, tr) = job
        c = ctx[t, f]
        p = dict(tr['parameters'], base_score=float(y[c['train']].mean()), eval_metric='rmse', n_jobs=1)
        if p['objective'] == 'reg:pseudohubererror':
            p['huber_slope'] = 2.0
        xx = matrices[name]
        m = xgb.XGBRegressor(**p)
        m.fit(xx[c['train']], y[c['train']])
        booster = m.get_booster()
        booster.feature_names = groups[name]
        matrix = xgb.DMatrix(xx[c['test']], feature_names=groups[name])
        pred = booster.predict(matrix)
        con = booster.predict(matrix, pred_contribs=True)
        assert np.allclose(con.sum(axis=1), pred, atol=0.0001, rtol=1e-05)
        path = OUT / f'n_ge_{t}' / f'fold_{f:02d}' / name / f"trial_{tr['trial']}"
        path.mkdir(parents=True, exist_ok=True)
        m.save_model(path / 'model.json')
        jsave(path / 'parameters.json', p)
        return (t, f, name, tr['trial'], pred, con)
    fitted = {}
    with ThreadPoolExecutor(max_workers=NWORK) as pool:
        for ft in as_completed([pool.submit(refit, j) for j in refitjobs]):
            (t, f, name, tr, p, c) = ft.result()
            fitted[t, f, name, tr] = (p, c)
    comparison = []
    foldrows = []
    rankrows = []
    for t in THRESHOLDS:
        ix = np.flatnonzero(support >= t)
        store = {n: (np.full(636, np.nan), np.full((636, len(fs) + 1), np.nan)) for (n, fs) in {**groups, 'training_only_selected': full}.items()}
        for ch in [z for z in choices if z['threshold'] == t]:
            f = ch['fold']
            test = ctx[t, f]['test']
            local = {}
            for (name, info) in ch['groups'].items():
                fit = [fitted[t, f, name, tr['trial']] for tr in info['trials']]
                if info['variant'] == 'best_single':
                    (p, c) = fit[0]
                else:
                    p = np.mean([p for (p, c) in fit], axis=0)
                    c = np.mean([c for (p, c) in fit], axis=0)
                    if info['variant'] == 'top3_calibrated':
                        (a, b) = info['calibration'].values()
                        p = a + b * p
                        c = b * c
                        c[:, -1] += a
                store[name][0][test] = p
                store[name][1][test] = c
                local[name] = (p, c)
                foldrows.append(dict(minimum_valid_pixels=t, group=name, fold=f, **metric(y[test], p)))
            name = ch['training_only_selected_group']
            (p, c) = local[name]
            fullc = np.zeros((len(test), len(full) + 1))
            fullc[:, [full.index(k) for k in groups[name]]] = c[:, :-1]
            fullc[:, -1] = c[:, -1]
            store['training_only_selected'][0][test] = p
            store['training_only_selected'][1][test] = fullc
            foldrows.append(dict(minimum_valid_pixels=t, group='training_only_selected', fold=f, selected_group=name, **metric(y[test], p)))
        for (name, (p, c)) in store.items():
            fs = full if name == 'training_only_selected' else groups[name]
            p = p[ix]
            c = c[ix]
            assert np.isfinite(p).all() and np.isfinite(c).all() and np.allclose(c.sum(axis=1), p, atol=0.0001, rtol=1e-05)
            path = OUT / f'n_ge_{t}' / name
            path.mkdir(exist_ok=True)
            pd.DataFrame(dict(city_id=d.city_id.iloc[ix].to_numpy(), observed_ST=y[ix], predicted_ST=p, outer_fold=outer[ix], actual_valid_ST_pixels=support[ix], sample_role='out_of_fold')).to_csv(path / 'OOF_predictions.csv', index=False, encoding='utf-8-sig')
            np.savez_compressed(path / 'OOF_SHAP.npz', feature_names=np.array(fs), feature_values=d[fs].to_numpy(np.float32)[ix], city_id=d.city_id.iloc[ix].to_numpy(str), shap_values=c[:, :-1], expected_value=c[:, -1], outer_fold=outer[ix])
            imp = pd.DataFrame(dict(feature=fs, mean_abs_shap=np.mean(abs(c[:, :-1]), axis=0)))
            imp['contribution_percent'] = 100 * imp.mean_abs_shap / imp.mean_abs_shap.sum()
            imp = imp.sort_values('mean_abs_shap', ascending=False)
            imp['rank'] = np.arange(1, len(imp) + 1)
            imp.to_csv(path / 'SHAP_ranking.csv', index=False, encoding='utf-8-sig')
            rankrows.append(imp.assign(minimum_valid_pixels=t, group=name))
            display = imp[~imp.feature.isin(['City_longitude', 'City_latitude'])]
            ranks = dict(zip(display.feature, range(1, len(display) + 1)))
            folds = pd.DataFrame([r for r in foldrows if r['minimum_valid_pixels'] == t and r['group'] == name])
            comparison.append(dict(minimum_valid_pixels=t, group=name, n_features=len(fs), **metric(y[ix], p), mean_fold_R2=float(folds.R2.mean()), sd_fold_R2=float(folds.R2.std(ddof=1)), solar_display_rank=ranks['Preseason_sos_srad'], wind_display_rank=ranks['Preseason_sos_ws'], BMSI_display_rank=ranks['BMSI_A']))
    result = pd.DataFrame(comparison)
    result.to_csv(OUT / 'screening_refit_model_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(foldrows).to_csv(OUT / 'fold_performance.csv', index=False, encoding='utf-8-sig')
    pd.concat(rankrows).to_csv(OUT / 'all_SHAP_ranks.csv', index=False, encoding='utf-8-sig')
    print('SCREEN_COMPLETE', result.to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
