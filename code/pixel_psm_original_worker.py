                                                                        
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
from pathlib import Path
import sys, json, time, calendar, hashlib, traceback, warnings
R = Path(os.environ['MORPH_PIXEL_PSM_WORKSPACE']).resolve()
sys.path.insert(0, str(R / 'runtime_deps'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(R / 'mplconfig')
import numpy as np
import pandas as pd
IN = R / 'input'
OUT = R / 'output'
CACHE = R / 'cache'
for p in [OUT, CACHE, R / 'jobs', R / 'done', R / 'logs']:
    p.mkdir(exist_ok=True)
GROUPS = ['OPL_months', 'end_month']
COV = ['background_DSR_monthly_mean_Wm2', 'ESA_tree_grass_pct', 'pixel_DEM_m', 'Preseason_sos_ppt', 'Preseason_sos_vpd', 'pixel_preseason_background_wind_ms']
SPECS = {'e': ('pixel_BMSI_A', 'received_direct_DSR_proxy_Wm2', COV + ['lat', 'lon', 'Preseason_sos_tmean']), 'f': ('received_direct_DSR_proxy_Wm2', 'lst_day_c', COV + ['pixel_BMSI_A', 'lat', 'lon', 'Preseason_sos_tmean'])}
SEM = 'Wind ~ BMSI\nShade ~ BMSI\nReceived ~ Shade + DSR\nDayLST ~ BMSI + Wind + Received\nST ~ BMSI + Wind + Received + DayLST\nBMSI ~~ DSR\nShade ~~ DSR\nWind ~~ DSR\nWind ~~ Shade'
COLS = {'BMSI': 'BMSI_A', 'Wind': 'Preseason_sos_ws', 'Shade': 'city_shade', 'DSR': 'city_background_DSR', 'Received': 'city_received_radiation', 'DayLST': 'lst_day_c', 'ST': 'ST_OLS_days_per_C'}

def save(p, x):
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(x, indent=2, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda : f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def pixels():
    return pd.read_csv(IN / 'full_pixels.csv.gz', dtype={'pixel_id': str}, float_precision='round_trip')

def job(name, kind, args=None, memory=2, priority=30):
    p = R / 'jobs' / f'{name}.json'
    if not p.exists():
        save(p, dict(name=name, kind=kind, args=args or {}, memory_GiB=memory, priority=priority))

def spatial_index():
    import rasterio
    q = pixels()
    coords = q[['lon', 'lat']].to_numpy()
    shared = Path('/lablibrary1/MOD11A2_1km_8day/monthly_aligned_500m')
    with rasterio.open(shared / 'LST_2001_04_500m.vrt') as src:
        (rr, cc) = rasterio.transform.rowcol(src.transform, coords[:, 0], coords[:, 1])
        rr = np.asarray(rr)
        cc = np.asarray(cc)
        good = (rr >= 0) & (cc >= 0) & (rr < src.height) & (cc < src.width)
        assert good.all(), 'Pixels outside the frozen source VRT grid'
        (nr, nc) = (src.height, src.width)
        meta = dict(height=nr, width=nc, transform=list(src.transform), crs=str(src.crs))
    block = 256
    code = rr // block * ((nc + block - 1) // block) + cc // block
    order = np.argsort(code, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(code[order])) + 1, len(order)]
    np.savez(CACHE / 'spatial_index.npz', row=rr, col=cc, order=order, boundaries=boundaries, source_index=q.source_index.to_numpy())
    save(OUT / 'LST_grid.json', dict(**meta, block_size=block, occupied_blocks=len(boundaries) - 1, n_pixels=len(q), source_root=str(shared)))
    print('spatial index', len(q), len(boundaries) - 1, flush=True)

def lst_month(key, tag='', max_blocks=None, validation=True):
    import rasterio
    from rasterio.windows import Window
    import xml.etree.ElementTree as ET
    shared = Path('/lablibrary1/MOD11A2_1km_8day/monthly_aligned_500m')
    source = shared / f'LST_{key}_500m.vrt'
    tree = ET.parse(source)
    for el in tree.iter('SourceFilename'):
        rel = el.text.replace('\\', '/')
        p = (shared / rel).resolve()
        if not p.is_relative_to(shared):
            raise ValueError('VRT source outside approved read-only data root')
        el.text = str(p)
        el.set('relativeToVRT', '0')
    vrtdir = CACHE / 'vrt'
    vrtdir.mkdir(exist_ok=True)
    vrt = vrtdir / f'{key}{tag}.vrt'
    tree.write(vrt, encoding='utf8')
    z = np.load(CACHE / 'spatial_index.npz')
    rr = z['row']
    cc = z['col']
    order = z['order']
    bounds = z['boundaries']
    vals = np.full(len(rr), np.nan, np.float32)
    nblocks = len(bounds) - 1
    selected = np.arange(nblocks) if max_blocks is None else np.unique(np.linspace(0, nblocks - 1, min(max_blocks, nblocks)).astype(int))
    start = time.monotonic()
    with rasterio.Env(GDAL_CACHEMAX=96 * 1024 ** 2, GDAL_NUM_THREADS='1'):
        with rasterio.open(vrt) as src:
            assert src.descriptions[0] == 'monthly_mean_LST_Day_C'
            for ib in selected:
                ix = order[bounds[ib]:bounds[ib + 1]]
                r0 = int(rr[ix[0]] // 256 * 256)
                c0 = int(cc[ix[0]] // 256 * 256)
                arr = src.read(1, window=Window(c0, r0, min(256, src.width - c0), min(256, src.height - r0)))
                v = arr[rr[ix] - r0, cc[ix] - c0].astype(np.float32)
                v[~np.isfinite(v) | (v < -100) | (v > 100)] = np.nan
                if src.nodata is not None:
                    v[v == src.nodata] = np.nan
                vals[ix] = v
    result = dict(month=key, seconds=time.monotonic() - start, blocks=len(selected), finite=int(np.isfinite(vals).sum()))
    if validation and max_blocks is None:
        chk = pd.read_csv(IN / 'lst_validation_order.csv')
        ix = pd.Index(z['source_index']).get_indexer(chk.source_index)
        assert (ix >= 0).all()
        with np.load(IN / 'lst_validation_values.npz') as old:
            expected = old[key].astype(float)
        expected[(expected < -100) | (expected > 100)] = np.nan
        observed = vals[ix]
        mask = np.isfinite(observed) & np.isfinite(expected)
        missing_mismatch = int(np.sum(np.isfinite(observed) != np.isfinite(expected)))
        error = float(np.max(np.abs(observed[mask] - expected[mask]))) if mask.any() else 0
        result.update(validation_n=len(chk), validation_missing_mismatches=missing_mismatch, validation_max_abs_error=error)
        assert missing_mismatch == 0 and error < 0.0001, result
    if not tag:
        md = CACHE / 'monthly'
        md.mkdir(exist_ok=True)
        np.save(md / f'{key}.npy', vals)
    return result

def aggregate_lst():
    from pyproj import Transformer
    q = pixels()
    n = len(q)
    result = np.full(n, np.nan)
    observations = np.zeros(n, int)
    minyears = np.zeros(n, int)
    nmonth = np.zeros(n, int)
    monthly = {p.stem: np.load(p, mmap_mode='r') for p in (CACHE / 'monthly').glob('*.npy')}
    assert len(monthly) == 204
    for ((opl, end), g) in q.groupby(GROUPS, sort=False):
        ix = g.index.to_numpy()
        means = []
        normal = []
        counts = []
        for offset in range(int(opl) - 1, -1, -1):
            total = np.zeros(len(ix))
            weight = np.zeros(len(ix))
            count = np.zeros(len(ix), int)
            days_list = []
            for year in range(2001, 2021):
                (yy, mm) = divmod(year * 12 + int(end) - 1 - offset, 12)
                mm += 1
                days = calendar.monthrange(yy, mm)[1]
                days_list.append(days)
                key = f'{yy:04d}_{mm:02d}'
                v = monthly[key][ix].astype(float) if key in monthly else np.full(len(ix), np.nan)
                good = np.isfinite(v)
                total[good] += v[good] * days
                weight[good] += days
                count += good
            means.append(np.divide(total, weight, out=np.full(len(ix), np.nan), where=weight > 0))
            normal.append(np.mean(days_list))
            counts.append(count)
        means = np.array(means)
        days = np.array(normal)[:, None]
        counts = np.array(counts)
        result[ix] = np.sum(means * days, axis=0) / days.sum()
        observations[ix] = counts.sum(axis=0)
        minyears[ix] = counts.min(axis=0)
        nmonth[ix] = (counts > 0).sum(axis=0)
    tr = Transformer.from_crs('EPSG:4326', '+proj=sinu +R=6371007.181 +nadgrids=@null +wktext', always_xy=True)
    (x, y) = tr.transform(q.lon.to_numpy(), q.lat.to_numpy())
    rr = np.floor((10007554.677 - y) / 926.625433055833).astype(int)
    cc = np.floor((x + 20015109.354) / 926.625433055833).astype(int)
    q['lst_day_c'] = result
    q['native_1km_cell'] = [f'{a}_{b}' for (a, b) in zip(rr, cc)]
    cols = ['source_index', 'pixel_id', 'city_id', 'OPL_months', 'end_month', 'lst_day_c', 'native_1km_cell']
    d = q[cols].copy()
    d['valid_day_month_year_values'] = observations
    d['minimum_years_per_OPL_month'] = minyears
    d['n_OPL_months_with_any_day_LST'] = nmonth
    d.to_csv(OUT / 'full_pixel_dayLST.csv.gz', index=False)
    q.to_csv(CACHE / 'full_pixels_LST.csv.gz', index=False)
    old = pd.read_csv(IN / 'old50_corrected.csv.gz', dtype={'pixel_id': str}, float_precision='round_trip')
    check = d[['pixel_id', 'lst_day_c']].merge(old[['pixel_id', 'lst_day_c']], on='pixel_id', suffixes=('_new', '_old'))
    a = check.lst_day_c_new.to_numpy()
    b = check.lst_day_c_old.to_numpy()
    good = np.isfinite(a) & np.isfinite(b)
    audit = dict(n_pixels=n, finite_LST=int(np.isfinite(result).sum()), shared_old_pixels=len(check), missing_mismatches=int(np.sum(np.isfinite(a) != np.isfinite(b))), max_abs_error=float(np.max(np.abs(a[good] - b[good]))))
    assert audit['missing_mismatches'] == 0 and audit['max_abs_error'] < 0.0001, audit
    return audit

def psm_prepare(which, baseline=False):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    label = ('baseline_' if baseline else '') + which
    folder = OUT / 'PSM' / label
    folder.mkdir(parents=True, exist_ok=True)
    if baseline:
        q = pd.read_csv(IN / 'old50_corrected.csv.gz', dtype={'pixel_id': str}, float_precision='round_trip')
    elif which == 'e':
        q = pixels()
        q['native_1km_cell'] = ''
    else:
        q = pd.read_csv(CACHE / 'full_pixels_LST.csv.gz', dtype={'pixel_id': str}, float_precision='round_trip')
    (driver, target, covars) = SPECS[which]
    d = q.replace([np.inf, -np.inf], np.nan).dropna(subset=[driver, target, *covars]).copy()
    d['cut'] = d.groupby(GROUPS)[driver].transform('median')
    d['treat'] = (d[driver] > d.cut).astype(int)
    d = d[d.groupby(GROUPS).treat.transform('nunique').eq(2)].reset_index(drop=True)
    scaler = StandardScaler().fit(d[covars])
    z = scaler.transform(d[covars])
    seasons = sparse.csr_matrix(pd.get_dummies(d[GROUPS].astype(str), drop_first=True).to_numpy(float))
    X = sparse.hstack([sparse.csr_matrix(z), seasons], format='csr')
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1500).fit(X, d.treat)
    assert clf.n_iter_[0] < 1500, 'Logistic fit did not converge'
    d['PS'] = clf.predict_proba(X)[:, 1]
    d['PS_logit'] = clf.decision_function(X)
    caliper = 0.1 * d.PS_logit.std(ddof=1)
    keep = list(dict.fromkeys(['city_id', 'source_index', 'pixel_id', *GROUPS, driver, target, *covars, 'native_1km_cell', 'cut', 'treat', 'PS', 'PS_logit']))
    d[keep].to_csv(folder / 'candidates.csv.gz', index=False)
    wd = CACHE / 'PSM' / label
    wd.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ((opl, end), g) in d.groupby(GROUPS, sort=False):
        hi = g[g.treat.eq(1)]
        lo = g[g.treat.eq(0)]
        key = f'{int(opl)}_{int(end)}'
        name = f'match_{label}_{key}'
        np.savez(wd / f'{key}.npz', hi_index=hi.index.to_numpy(), lo_index=lo.index.to_numpy(), hi_logit=hi.PS_logit.to_numpy(), lo_logit=lo.PS_logit.to_numpy(), caliper=caliper)
        matrix = len(hi) * len(lo) * 8 / 1024 ** 3
        job(name, 'match', dict(label=label, key=key), memory=max(2, 1.25 * matrix + 2), priority=20 if baseline else 25)
        jobs.append(dict(name=name, key=key, n_high=len(hi), n_low=len(lo), matrix_GiB=matrix))
    feature_names = covars + pd.get_dummies(d[GROUPS].astype(str), drop_first=True).columns.tolist()
    np.savez_compressed(folder / 'propensity_model.npz', coefficients=clf.coef_, intercept=clf.intercept_, mean=scaler.mean_, scale=scaler.scale_, caliper=np.asarray([caliper]), features=np.asarray(feature_names))
    return dict(label=label, n_candidates=len(d), n_candidate_cities=int(d.city_id.nunique()), caliper_logit=float(caliper))

def match(label, key):
    from scipy.optimize import linear_sum_assignment
    z = np.load(CACHE / 'PSM' / label / f'{key}.npz')
    a = z['hi_logit']
    b = z['lo_logit']
    caliper = float(z['caliper'])
    assert len(a) <= len(b), 'Expected high group no larger than low group'
    matrix = np.empty((len(a), len(b)), dtype=np.float64)
    chunk = max(1, min(256, 32 * 1024 ** 2 // max(8, len(b) * 8)))
    for i in range(0, len(a), chunk):
        v = matrix[i:i + chunk]
        np.subtract(a[i:i + chunk, None], b[None, :], out=v)
        np.abs(v, out=v)
        v[v > caliper] = 1000000.0
    started = time.monotonic()
    (ih, il) = linear_sum_assignment(matrix)
    delta = np.abs(a[ih] - b[il])
    good = delta <= caliper
    folder = OUT / 'PSM' / label / 'windows'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f'{key}.npz', high=z['hi_index'][ih[good]], low=z['lo_index'][il[good]], distance=delta[good])
    return dict(label=label, key=key, pairs=int(good.sum()), solver_seconds=time.monotonic() - started, matrix_GiB=matrix.nbytes / 1024 ** 3)

def effect(p):
    from scipy.stats import t
    values = p.delta_outcome.to_numpy()
    estimate = float(values.mean())
    score = values - estimate
    a = p.high_city_id.to_numpy()
    b = p.low_city_id.to_numpy()
    cities = np.unique(np.r_[a, b])
    node = {int(c): 0.0 for c in cities}
    dyad = {}
    for (s, i, j) in zip(score, a, b):
        node[i] += s
        if i != j:
            node[j] += s
            key = tuple(sorted((i, j)))
            dyad[key] = dyad.get(key, 0.0) + s
    meat = sum((v * v for v in node.values())) - sum((v * v for v in dyad.values()))
    if meat <= 0:
        raise ValueError('Nonpositive dyadic variance; inspect rather than clamp')
    se = float(np.sqrt(meat * len(cities) / (len(cities) - 1)) / len(values))
    crit = t.ppf(0.975, len(cities) - 1)
    return dict(estimate=estimate, SE_dyadic_city_cluster=se, CI95_dyadic_city_cluster=[estimate - crit * se, estimate + crit * se], P_dyadic_city_cluster=float(2 * t.sf(abs(estimate / se), len(cities) - 1)), n_pairs=len(p), n_cities=len(cities), n_cross_city_pairs=int(np.sum(a != b)))

def psm_summary(label):
    folder = OUT / 'PSM' / label
    which = label[len('baseline_'):] if label.startswith('baseline_') else label
    baseline = label.startswith('baseline_')
    (driver, target, cov) = SPECS[which]
    with np.load(folder / 'propensity_model.npz') as fitted_model:
        caliper = float(fitted_model['caliper'][0])
    d = pd.read_csv(folder / 'candidates.csv.gz', dtype={'pixel_id': str, 'native_1km_cell': str}, float_precision='round_trip')
    high = []
    low = []
    dist = []
    for ((opl, end), group) in d.groupby(GROUPS, sort=True):
        key = f'{int(opl)}_{int(end)}'
        with np.load(folder / 'windows' / f'{key}.npz') as z:
            high.extend(z['high'])
            low.extend(z['low'])
            dist.extend(z['distance'])
    hi = d.loc[high].reset_index(drop=True)
    lo = d.loc[low].reset_index(drop=True)
    p = pd.DataFrame(dict(high_city_id=hi.city_id, low_city_id=lo.city_id, high_pixel_id=hi.pixel_id, low_pixel_id=lo.pixel_id, high_driver=hi[driver], low_driver=lo[driver], high_outcome=hi[target], low_outcome=lo[target], delta_outcome=hi[target] - lo[target], PS_logit_distance=dist))
    for k in GROUPS:
        p[k] = hi[k].to_numpy()
        assert np.array_equal(hi[k], lo[k])
    for k in cov:
        p['high_' + k] = hi[k].to_numpy()
        p['low_' + k] = lo[k].to_numpy()
    p['same_nominal_1km_cell'] = hi.native_1km_cell.eq(lo.native_1km_cell) & hi.native_1km_cell.notna()
    assert p.high_pixel_id.is_unique and p.low_pixel_id.is_unique and (not set(p.high_pixel_id) & set(p.low_pixel_id))
    assert p.high_driver.gt(p.low_driver).all() and p.PS_logit_distance.le(caliper).all()
    p.to_csv(folder / 'pairs.csv.gz', index=False)
    bal = []
    for k in cov:
        centre = d[k] - d.groupby(GROUPS)[k].transform('mean')
        for (style, values) in [('global', d[k]), ('within_exact_stratum', centre)]:
            h = values[d.treat.eq(1)].to_numpy()
            l = values[d.treat.eq(0)].to_numpy()
            sd = np.sqrt((h.var(ddof=1) + l.var(ddof=1)) / 2)
            bal.append(dict(covariate=k, scope=style, SMD_before=float((h.mean() - l.mean()) / sd) if sd > 1e-12 else 0, SMD_after=float((values.loc[high].mean() - values.loc[low].mean()) / sd) if sd > 1e-12 else 0, fixed_prematch_pooled_SD=float(sd)))
    b = pd.DataFrame(bal)
    b.to_csv(folder / 'balance.csv', index=False)
    stats = effect(p)
    stats.update(n_candidates=len(d), driver=driver, outcome=target, caliper_logit=caliper, max_abs_SMD_after_global=float(b[b.scope.eq('global')].SMD_after.abs().max()), max_abs_SMD_after_within_stratum=float(b[b.scope.eq('within_exact_stratum')].SMD_after.abs().max()), same_nominal_1km_pairs=int(p.same_nominal_1km_cell.sum()))
    if which == 'f':
        stats['different_nominal_1km_cells_sensitivity'] = effect(p[~p.same_nominal_1km_cell])
    if baseline:
        old = json.loads((IN / f"old_{('e_BMSI_received' if which == 'e' else 'f_received_dayLST')}_summary.json").read_text(encoding='utf-8-sig'))
        stats['baseline_n_pair_difference'] = stats['n_pairs'] - old['n_pairs']
        stats['baseline_estimate_difference'] = stats['estimate'] - old['estimate']
        assert stats['baseline_n_pair_difference'] == 0 and abs(stats['baseline_estimate_difference']) < 1e-09, stats
    save(folder / 'summary.json', stats)
    return stats

def sem_fit(d, detail=False):
    from semopy import Model
    a = d[list(COLS)].astype(float)
    z = (a - a.mean()) / a.std(ddof=0)
    m = Model(SEM)
    fit = m.fit(z, obj='MLW')
    if not fit.success:
        raise ValueError('SEM did not converge')
    p = m.inspect(std_est=True, information='expected' if detail else None)
    v = {(r.rval, r.lval): r['Est. Std'] for (_, r) in p[p.op == '~'].iterrows()}
    bs = v['BMSI', 'Shade']
    sr = v['Shade', 'Received']
    rl = v['Received', 'DayLST']
    lt = v['DayLST', 'ST']
    bw = v['BMSI', 'Wind']
    wl = v['Wind', 'DayLST']
    bl = v['BMSI', 'DayLST']
    effects = dict(ST_from_radiation_via_LST=rl * lt, ST_via_shade_radiation_LST=bs * sr * rl * lt, ST_via_wind_LST=bw * wl * lt, LST_via_shade=bs * sr * rl, LST_via_wind=bw * wl, LST_direct=bl, ST_direct=v['BMSI', 'ST'], ST_via_LST_direct=bl * lt, ST_via_wind=bw * v['Wind', 'ST'], ST_via_shade_radiation=bs * sr * v['Received', 'ST'])
    effects['ST_sum'] = sum((effects[k] for k in ['ST_direct', 'ST_via_LST_direct', 'ST_via_wind', 'ST_via_shade_radiation', 'ST_via_shade_radiation_LST', 'ST_via_wind_LST']))
    if not detail:
        return effects
    return (effects, p)


def sem_prepare():
    q = pd.read_csv(CACHE / 'full_pixels_LST.csv.gz', dtype={'pixel_id': str}, float_precision='round_trip')
    raw = pd.read_csv(IN / 'old_city_sem.csv')
    old = pd.read_csv(IN / 'old50_corrected.csv.gz', dtype={'pixel_id': str}, float_precision='round_trip')
    fields = {'shade_fraction': 'city_shade', 'background_DSR_monthly_mean_Wm2': 'city_background_DSR', 'received_direct_DSR_proxy_Wm2': 'city_received_radiation', 'lst_day_c': 'lst_day_c'}

    def assemble(p, weights=None):
        rows = []
        for (cid, g) in p.groupby('city_id'):
            row = {'city_id': cid, 'eligible_pixels': len(g), 'LST_valid_pixels': int(g.lst_day_c.notna().sum())}
            for (src, dst) in fields.items():
                valid = g[src].notna()
                w = np.ones(len(g)) if weights is None else g[weights].to_numpy()
                row[dst] = float(np.average(g.loc[valid, src], weights=w[valid])) if valid.any() else np.nan
            rows.append(row)
        agg = pd.DataFrame(rows)
        d = raw.drop(columns=list(fields.values())).merge(agg, on='city_id', how='inner')
        return d
    variants = {'legacy_city_inputs': raw, 'old50_corrected_window': assemble(old, 'pixel_city_weight')}
    sub = q[q.pixel_id.isin(old.pixel_id)].copy()
    pop = q.groupby(['city_id', 'sample_stratum']).size()
    ns = sub.groupby(['city_id', 'sample_stratum']).size()
    sub['sample_weight'] = [pop.loc[c, s] / ns.loc[c, s] for (c, s) in zip(sub.city_id, sub.sample_stratum)]
    variants['old50_new_receivers'] = assemble(sub, 'sample_weight')
    variants['full_field'] = assemble(q)
    folder = OUT / 'SEM'
    folder.mkdir(exist_ok=True)
    for (name, d) in variants.items():
        keep = ['city_id', 'lon_gpkg', 'lat_gpkg', *COLS.values()]
        d = d[keep].rename(columns={v: k for (k, v) in COLS.items()}).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        d.to_csv(folder / f'{name}_input.csv', index=False)
        variants[name] = d
        (eff, p) = sem_fit(d, True)
        p.to_csv(folder / f'{name}_paths.csv', index=False)
        save(folder / f'{name}_effects.json', eff)
    common = set.intersection(*(set(d.city_id) for d in variants.values()))
    rows = []
    for (name, d) in variants.items():
        (eff, p) = sem_fit(d[d.city_id.isin(common)], True)
        p.to_csv(folder / f'{name}_common_paths.csv', index=False)
        rows.append(dict(variant=name, n_cities=int(d.city_id.isin(common).sum()), **eff))
    pd.DataFrame(rows).to_csv(folder / 'common_city_comparison.csv', index=False)
    for size in [5, 10]:
        for start in range(0, 1000, 25):
            job(f'sem_boot_{size}_{start}', 'sem_boot', dict(size=size, start=start, stop=start + 25), memory=1.5, priority=30)
    return dict(common_cities=len(common), variant_n={k: len(v) for (k, v) in variants.items()})

def sem_boot(size, start, stop):
    d = pd.read_csv(OUT / 'SEM/full_field_input.csv')
    blocks = np.floor((d[['lon_gpkg', 'lat_gpkg']].to_numpy() + [180, 90]) / size).astype(int)
    (_, group) = np.unique(blocks, axis=0, return_inverse=True)
    u = np.unique(group)
    lookup = [np.flatnonzero(group == g) for g in u]
    rows = []
    for i in range(start, stop):
        rng = np.random.default_rng(20261024 + i * 7919 + size)
        ix = np.concatenate([lookup[g] for g in rng.choice(u, len(u), replace=True)])
        try:
            result = dict(replicate=i, **sem_fit(d.iloc[ix]))
        except Exception as exc:
            result = dict(replicate=i, error=str(exc))
        rows.append(result)
    folder = OUT / 'SEM/bootstrap'
    folder.mkdir(exist_ok=True)
    save(folder / f'{size}_{start}.json', rows)
    return dict(size=size, start=start, successful=sum(('error' not in r for r in rows)), attempted=len(rows))

def sem_summary():
    effects = json.loads((OUT / 'SEM/full_field_effects.json').read_text())
    rows = []
    for size in [5, 10]:
        values = []
        for start in range(0, 1000, 25):
            values.extend(json.loads((OUT / 'SEM/bootstrap' / f'{size}_{start}.json').read_text()))
        good = pd.DataFrame([v for v in values if 'error' not in v])
        good.to_csv(OUT / 'SEM' / f'bootstrap_{size}deg.csv', index=False)
        for (path, estimate) in effects.items():
            rows.append(dict(path=path, block_degrees=size, estimate=estimate, ci_low=good[path].quantile(0.025), ci_high=good[path].quantile(0.975), successful=len(good), attempted=len(values)))
    pd.DataFrame(rows).to_csv(OUT / 'SEM/indirect_effects_CI.csv', index=False)
    return dict(successful_by_block_degrees={s: next((r['successful'] for r in rows if r['block_degrees'] == s)) for s in [5, 10]})

def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    name = spec['name']
    start = time.time()
    kind = spec['kind']
    args = spec.get('args', {})
    import resource
    limit = int((spec.get('memory_GiB', 2) + 2) * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    funcs = {'index': spatial_index, 'lst': lst_month, 'aggregate_lst': aggregate_lst, 'psm_prepare': psm_prepare, 'match': match, 'psm_summary': psm_summary, 'sem_prepare': sem_prepare, 'sem_boot': sem_boot, 'sem_summary': sem_summary}
    try:
        result = funcs[kind](**args)
    except Exception:
        traceback.print_exc()
        raise
if __name__ == '__main__':
    main()
