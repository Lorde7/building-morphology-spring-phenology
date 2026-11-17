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
import os, json, hashlib
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
root = Path(configured_path('experiment_figure3_station_matching.path1'))
source = Path(configured_path('experiment_figure3_station_matching.path2'))
d = pd.read_csv(source)
assert len(d) == 1177 and d.station_key.is_unique
paths = {'ppt_dem': root / 'data/delivery/station_covariates_v2.csv', 'vpd': root / 'data/delivery/station_vpd.csv', 'wind': root / 'data/root_background_wind/station_background_wind.csv', 'cover': root / 'data/root_worldcover/station_worldcover2020.csv'}
for p in paths.values():
    assert p.exists(), p
features = [('ppt_dem', ['preseason_ppt', 'elevation_dem_native_point']), ('vpd', ['preseason_vpd_kPa', 'complete_years']), ('wind', ['background_preseason_wind_ms', 'background_wind_complete_years']), ('cover', ['ESA_tree_grass_pct', 'worldcover_valid_cells'])]
for (name, cols) in features:
    f = pd.read_csv(paths[name])
    assert len(f) == 1177 and f.station_key.is_unique and (set(f.station_key) == set(d.station_key))
    d = d.merge(f[['station_key'] + cols], on='station_key', validate='one_to_one', sort=False)
assert d.station_key.tolist() == pd.read_csv(source).station_key.tolist()
assert d.ESA_tree_grass_pct.dropna().between(0, 100).all()
assert d.preseason_vpd_kPa.dropna().between(0, 20).all()
assert d.background_preseason_wind_ms.dropna().between(0, 100).all()
assert d.preseason_ppt.dropna().ge(0).all()
assert (d.preseason_vpd_kPa.isna() | d.complete_years.ge(1)).all()
assert (d.background_preseason_wind_ms.isna() | d.background_wind_complete_years.ge(1)).all()
assert np.array_equal(d['treat'].to_numpy(), (d.bmsi > d.source_median_BMSI).astype(int).to_numpy())
oldcov = ['latitude', 'longitude', 'background_DSR_mean_W_m2', 'OPL', 'n_complete_OPL_wind_years', 'height_log', 'Ta_OPL_mean_C']
newcov = ['latitude', 'longitude', 'background_DSR_mean_W_m2', 'n_complete_OPL_wind_years', 'height_log', 'Ta_OPL_mean_C', 'ESA_tree_grass_pct', 'elevation_dem_native_point', 'preseason_ppt', 'preseason_vpd_kPa', 'background_preseason_wind_ms']
assert len(newcov) == 11 and 'OPL' not in newcov
complete = np.isfinite(d[newcov].to_numpy(float)).all(axis=1)
cohort = d.loc[complete].copy()
assert not cohort.empty
exact = ['source', 'end_month', 'height_known']
out = root / 'analysis/expanded_matching'
out.mkdir(exist_ok=False)
d.to_csv(out / 'all_candidates_with_covariates.csv', index=False, encoding='utf-8-sig')
cohort[['station_key', 'source', 'treat']].to_csv(out / 'frozen_complete_case_keys.csv', index=False)
missing = {c: {'n': int(d[c].isna().sum()), 'keys': d.loc[d[c].isna(), 'station_key'].tolist()} for c in newcov}

def fit(name, q, covariates):
    target = out / name
    target.mkdir()
    q = q.copy()
    z = (q[covariates] - q.groupby('source')[covariates].transform('mean')) / q.groupby('source')[covariates].transform('std').replace(0, 1).fillna(1)
    X = pd.concat([z, pd.get_dummies(q.source, prefix='source', drop_first=True), pd.get_dummies(q.end_month.astype(int), prefix='end_month', drop_first=True), q[['height_known']]], axis=1).astype(float)
    assert np.isfinite(X.to_numpy()).all()
    if name == 'expanded_no_OPL':
        assert not any(('OPL' == str(c) or str(c).startswith('OPL_') for c in X.columns))
    clf = LogisticRegression(C=1, max_iter=1000, solver='lbfgs').fit(X, q.treat)
    assert clf.n_iter_[0] < 1000
    q['PS'] = clf.predict_proba(X)[:, 1]
    q['logit'] = clf.decision_function(X)
    caliper = 0.2 * q.logit.std(ddof=1)
    pairs = []
    strata = []
    for (group, g) in q.groupby(exact):
        hi = g[g.treat.eq(1)]
        lo = g[g.treat.eq(0)]
        n0 = len(pairs)
        if not hi.empty and (not lo.empty):
            distance = np.abs(hi.logit.to_numpy()[:, None] - lo.logit.to_numpy()[None, :])
            (ii, jj) = linear_sum_assignment(np.where(distance <= caliper, distance, 1000000.0))
            for (i, j) in zip(ii, jj):
                if distance[i, j] > caliper:
                    continue
                h = hi.iloc[i]
                l = lo.iloc[j]
                row = {'pair_id': len(pairs), 'source': group[0], 'end_month': group[1], 'height_known': group[2], 'high_station': h.station_key, 'low_station': l.station_key, 'high_pixel': h.pixel_cluster_id, 'low_pixel': l.pixel_cluster_id, 'high_wind': h.wind_OPL_mean_m_s, 'low_wind': l.wind_OPL_mean_m_s, 'delta_wind': h.wind_OPL_mean_m_s - l.wind_OPL_mean_m_s, 'high_BMSI': h.bmsi, 'low_BMSI': l.bmsi, 'high_PS': h.PS, 'low_PS': l.PS, 'PS_logit_distance': float(distance[i, j]), 'high_OPL_descriptive_only': h.OPL, 'low_OPL_descriptive_only': l.OPL}
                for c in newcov + ['OPL']:
                    row['high_' + c] = h[c]
                    row['low_' + c] = l[c]
                pairs.append(row)
        strata.append({'source': group[0], 'end_month': group[1], 'height_known': group[2], 'high_available': len(hi), 'low_available': len(lo), 'pairs': len(pairs) - n0, 'no_comparison_group': hi.empty or lo.empty})
    p = pd.DataFrame(pairs)
    assert not p.empty
    assert p.high_station.is_unique and p.low_station.is_unique and (not set(p.high_station) & set(p.low_station))
    assert (p.PS_logit_distance <= caliper + 1e-12).all()
    lookup = q.set_index('station_key')
    for c in exact:
        assert np.array_equal(p.high_station.map(lookup[c]).to_numpy(), p.low_station.map(lookup[c]).to_numpy())
    parent = {}

    def find(v):
        parent.setdefault(v, v)
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]
    for (h, l) in zip(p.high_pixel, p.low_pixel):
        parent[find(h)] = find(l)
    p['pair_pixel_cluster'] = [find(v) for v in p.high_pixel]
    ncl = p.pair_pixel_cluster.nunique()
    assert ncl >= 2
    m = sm.OLS(p.delta_wind, np.ones((len(p), 1))).fit(cov_type='cluster', cov_kwds={'groups': p.pair_pixel_cluster})
    ci = m.conf_int().iloc[0].tolist()
    balance = []
    for c in newcov + ['OPL']:
        h = q.loc[q.treat.eq(1), c].to_numpy(float)
        l = q.loc[q.treat.eq(0), c].to_numpy(float)
        sd = float(np.sqrt((np.var(h, ddof=1) + np.var(l, ddof=1)) / 2))
        pre = float(h.mean() - l.mean())
        post = float((p['high_' + c] - p['low_' + c]).mean())
        if sd == 0 and (pre != 0 or post != 0):
            raise ValueError('Zero variance with group difference')
        balance.append({'covariate': c, 'entered_propensity': c in covariates, 'fixed_prematch_pooled_SD': sd, 'SMD_before': pre / sd if sd else 0.0, 'SMD_after': post / sd if sd else 0.0})
    b = pd.DataFrame(balance)
    result = {'name': name, 'n_candidates': len(q), 'n_high': int(q.treat.sum()), 'n_low': int(q.treat.eq(0).sum()), 'n_matched_pairs': len(p), 'n_matched_stations': 2 * len(p), 'n_pair_clusters': int(ncl), 'mean_high_minus_low_wind_ms': float(m.params.iloc[0]), 'CI95': ci, 'P': float(m.pvalues.iloc[0]), 'SE_cluster': float(m.bse.iloc[0]), 'caliper_logit': float(caliper), 'caliper_SD_fraction': 0.2, 'continuous_covariates': covariates, 'design_columns': list(X.columns), 'logistic_iterations': int(clf.n_iter_[0]), 'max_abs_SMD_after_entered': float(b.loc[b.entered_propensity, 'SMD_after'].abs().max()), 'above_0p1': b.loc[b.entered_propensity & b.SMD_after.abs().gt(0.1)].to_dict('records'), 'above_0p2': b.loc[b.entered_propensity & b.SMD_after.abs().gt(0.2)].to_dict('records'), 'sources': p.source.value_counts().to_dict(), 'OPL_pair_abs_difference_mean_descriptive_only': float(abs(p.high_OPL_descriptive_only - p.low_OPL_descriptive_only).mean())}
    p.to_csv(target / 'pairs.csv', index=False)
    b.to_csv(target / 'balance.csv', index=False)
    q.to_csv(target / 'candidates.csv', index=False)
    pd.DataFrame(strata).to_csv(target / 'strata.csv', index=False)
    (target / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return (result, p)
(baseline, p0) = fit('baseline_full_1177', d, oldcov)
assert baseline['n_matched_pairs'] == 463 and abs(baseline['mean_high_minus_low_wind_ms'] + 0.21521105725224515) < 1e-10
(old, p1) = fit('old_covariates_complete_cases', cohort, oldcov)
(new, p2) = fit('expanded_no_OPL', cohort, newcov)
assert set(pd.read_csv(out / 'old_covariates_complete_cases/candidates.csv').station_key) == set(pd.read_csv(out / 'expanded_no_OPL/candidates.csv').station_key)
pairset = lambda p: set(zip(p.high_station, p.low_station))
report = {'task': 'STATION-PSM-20261113', 'status': 'computed_pending_independent_review', 'baseline': baseline, 'old_on_complete': old, 'expanded_no_OPL': new, 'complete_cases': len(cohort), 'incomplete_cases': int((~complete).sum()), 'missing_by_covariate': missing, 'same_pairs_old_new': len(pairset(p1) & pairset(p2)), 'new_only_pairs': len(pairset(p2) - pairset(p1)), 'shared_matched_stations': len((set(p1.high_station) | set(p1.low_station)) & (set(p2.high_station) | set(p2.low_station))), 'input_hashes': {k: hashlib.sha256(p.read_bytes()).hexdigest() for (k, p) in paths.items()}, 'scientific_scope': 'conditional association, not BMSI total or direct causal effect; expanded specification and rematching considered together', 'manuscript_updated': False}
print(json.dumps({k: report[k] for k in ['status', 'complete_cases', 'incomplete_cases', 'same_pairs_old_new', 'shared_matched_stations']}, ensure_ascii=False))
for x in [baseline, old, new]:
    print(json.dumps({k: x[k] for k in ['name', 'n_candidates', 'n_matched_pairs', 'mean_high_minus_low_wind_ms', 'CI95', 'P', 'max_abs_SMD_after_entered', 'sources']}, ensure_ascii=False))
