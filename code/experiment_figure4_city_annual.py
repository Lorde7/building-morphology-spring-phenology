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
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import sys, json, time, shutil
import numpy as np, pandas as pd, rasterio, numba
import netCDF4
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.sparse import csr_matrix
from scipy.stats import t as student_t
from resample_model_ST_Innovation_20261012 import OUT as PREVIOUS, MODELROOT, RESULT, CITY, interp, weighted_city
from calculate_TRENDY_historical_ST_benchmark_20261010 import optimized, regress
from calculate_all20_model_ST_20261012 import common_predictors
OUT = PREVIOUS / 'city_annual_OLS_global_maps'
YEARS = np.arange(2001, 2021)
CLIMATE = Path(configured_path('experiment_figure4_city_annual.path1'))
PHENO = Path(configured_path('experiment_figure4_city_annual.path2'))
NATIVE_CACHE = Path(configured_path('experiment_figure4_city_annual.path3'))

def save(p, o):
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding='utf-8')

def pixels():
    a = dict(np.load(PREVIOUS / 'urban_pixel_observed_and_grid.npz'))
    keep = np.load(PREVIOUS / 'common_urban_pixel_mask.npy')
    q = {k: v[keep] if v.ndim == 1 and len(v) == len(keep) else v for (k, v) in a.items()}
    (q['cities'], q['city_index']) = np.unique(q['city_id'], return_inverse=True)
    return q

def blocked_values(path, q, native_month=None):
    rr = q['row']
    cc = q['col']
    v = np.full(len(rr), np.nan, np.float32)
    with rasterio.open(path) as original:
        if native_month is None:
            src = original
        else:
            with rasterio.open(CLIMATE / 'TerraClimate_tmean_2001_M01_aligned_mask.tif') as ref:
                src = WarpedVRT(original, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height, resampling=Resampling.bilinear, dtype='float32', nodata=np.nan)
        (bh, bw) = src.block_shapes[0]
        bid = rr // bh * int(np.ceil(src.width / bw)) + cc // bw
        order = np.argsort(bid)
        cut = np.r_[0, np.flatnonzero(np.diff(bid[order])) + 1, len(order)]
        for i in range(len(cut) - 1):
            ix = order[cut[i]:cut[i + 1]]
            r0 = rr[ix[0]] // bh * bh
            c0 = cc[ix[0]] // bw * bw
            band = 1 if native_month is None else native_month
            a = src.read(band, window=Window(c0, r0, min(bw, src.width - c0), min(bh, src.height - r0)), masked=True).astype(np.float32).filled(np.nan)
            v[ix] = a[rr[ix] - r0, cc[ix] - c0]
        band = 0 if native_month is None else native_month - 1
        v = v * original.scales[band] + original.offsets[band]
        if native_month is not None:
            src.close()
    return v

def native_values(path, q, month):
    with rasterio.open(path) as src:
        tr = src.transform
        shape = src.shape
    with netCDF4.Dataset(str(path)) as ds:
        variable = path.stem.split('_')[1]
        a = np.ma.asarray(ds.variables[variable][month - 1], dtype=np.float32).filled(np.nan)
        lat = np.asarray(ds.variables['lat'][:])
        lon = np.asarray(ds.variables['lon'][:])
        if lat[0] < lat[-1]:
            a = a[::-1]
        if lon[0] > lon[-1]:
            a = a[:, ::-1]
    x = (q['lon_exact'] - tr.c) / tr.a - 0.5
    y = (q['lat_exact'] - tr.f) / tr.e - 0.5
    c = np.floor(x).astype(int)
    r = np.floor(y).astype(int)
    dx = x - c
    dy = y - r
    numerator = np.zeros(len(x))
    denominator = np.zeros(len(x))
    for (dr, dc, w) in [(0, 0, (1 - dx) * (1 - dy)), (0, 1, dx * (1 - dy)), (1, 0, (1 - dx) * dy), (1, 1, dx * dy)]:
        v = a[np.clip(r + dr, 0, shape[0] - 1), np.clip(c + dc, 0, shape[1] - 1)]
        good = np.isfinite(v)
        numerator += np.where(good, v * w, 0)
        denominator += np.where(good, w, 0)
    result = np.divide(numerator, denominator, out=np.full(len(x), np.nan), where=denominator > 0)
    parent_r = np.floor((q['lat_exact'] - tr.f) / tr.e).astype(int)
    parent_c = np.floor((q['lon_exact'] - tr.c) / tr.a).astype(int)
    result[~np.isfinite(a[parent_r, parent_c])] = np.nan
    if month in [1, 10]:
        from rasterio.warp import reproject
        ids = np.random.default_rng(20261012).choice(len(x), 12, replace=False)
        check = []
        with rasterio.open(CLIMATE / 'TerraClimate_tmean_2001_M01_aligned_mask.tif') as ref:
            for i in ids:
                r0 = max(0, r[i] - 2)
                c0 = max(0, c[i] - 2)
                block = a[r0:r[i] + 4, c0:c[i] + 4]
                out = np.full((1, 1), np.nan, np.float32)
                reproject(block, out, src_transform=rasterio.windows.transform(Window(c0, r0, block.shape[1], block.shape[0]), tr), src_crs='EPSG:4326', src_nodata=np.nan, dst_transform=ref.window_transform(Window(int(q['col'][i]), int(q['row'][i]), 1, 1)), dst_crs=ref.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
                error = float(abs(out[0, 0] - result[i]))
                assert error < 0.0002, (i, error)
                check.append(dict(pixel_index=int(i), GDAL=float(out[0, 0]), direct=float(result[i]), difference=error))
    return result.astype(np.float32)

def observed_year(year):
    path = OUT / 'observed_annual_cache' / f'{year}.npz'
    if path.exists():
        return (year, 'cached')
    q = pixels()
    n = len(q['cities'])
    cubes = {v: np.full((9, n), np.nan) for v in ['tmean', 'ppt', 'srad']}
    for (m, (yr, mo)) in enumerate([(year - 1, 10), (year - 1, 11), (year - 1, 12)] + [(year, mo) for mo in range(1, 7)]):
        for v in cubes:
            original_path = CLIMATE / f'TerraClimate_{v}_{yr}_M{mo:02d}_aligned_mask.tif'
            if original_path.exists():
                vals = blocked_values(original_path, q)
            else:
                vals = native_values(NATIVE_CACHE / f'TerraClimate_{v}_{yr}.nc', q, mo)
            vals[vals < -1000] = np.nan
            cubes[v][m] = weighted_city(vals, q)
    vals = blocked_values(PHENO / f'sos_{year}.tif', q)
    vals[(vals < 1) | (vals >= 190)] = np.nan
    sos = weighted_city(vals, q)
    np.savez_compressed(path, cities=q['cities'], SOS=sos, **cubes)
    return (year, 'complete')

def weather_weights(q, keys):
    n = len(q['city_id'])
    index = {cell: i for (i, cell) in enumerate(keys)}
    lo = q['lon_exact']
    la = q['lat_exact']
    cx = (lo + 180) / 3 - 0.5
    ry = (90 - la) / 3 - 0.5
    c = np.floor(cx).astype(int)
    r = np.floor(ry).astype(int)
    dx = cx - c
    dy = ry - r
    den = np.zeros(n)
    parts = []
    for (dr, dc, w) in [(0, 0, (1 - dx) * (1 - dy)), (0, 1, dx * (1 - dy)), (1, 0, (1 - dx) * dy), (1, 1, dx * dy)]:
        ix = np.array([index.get(f'{i}:{j % 120}', -1) for (i, j) in zip(r + dr, c + dc)], dtype=np.int32)
        den += np.where(ix >= 0, w, 0)
        parts.append((ix, w))
    assert (den > 0).all()
    area = q['pixel_area_weight']
    tot = np.bincount(q['city_index'], weights=area)
    weights = np.zeros((len(q['cities']), len(keys)))
    for (ix, w) in parts:
        good = ix >= 0
        ww = w[good] / den[good] * area[good] / tot[q['city_index'][good]]
        np.add.at(weights, (q['city_index'][good], ix[good]), ww)
    assert np.max(abs(weights.sum(axis=1) - 1)) < 1e-10
    return weights

def model_series(family, model):
    path = OUT / 'model_annual_cache' / f'{model}.npz'
    if path.exists():
        return (family, model, 'cached')
    q = pixels()
    grid = pd.read_csv(MODELROOT / 'common_3degree_cells_containing_cities.csv', dtype={'cell_id': str})
    keys = grid.cell_id.tolist()
    folder = MODELROOT if family == 'CMIP6' else MODELROOT / 'TRENDY_common_3degree'
    raw = pd.read_csv(folder / 'model_SOS' / f'{model}.csv', dtype={'cell_id': str})
    yy = raw.pivot(index='year', columns='cell_id', values='SOS_15').reindex(index=YEARS, columns=keys).to_numpy(float)
    yy[(yy < 1) | (yy >= 190)] = np.nan
    sos = []
    for y in range(20):
        a = np.full((60, 120), np.nan, np.float32)
        for (cell, v) in zip(keys, yy[y]):
            (i, j) = map(int, cell.split(':'))
            a[i, j] = v
        vals = interp(a, q)
        vals[~np.isfinite(a[q['coarse_row'], q['coarse_col']])] = np.nan
        sos.append(weighted_city(vals, q))
    values = dict(SOS=np.asarray(sos), cities=q['cities'])
    if family == 'CMIP6':
        W = weather_weights(q, keys)
        for v in ['tas', 'pr', 'rsds']:
            df = pd.read_csv(MODELROOT / 'monthly_model_fields' / model / f'{v}.csv', dtype={'cell_id': str})
            series = df.pivot(index=['year', 'month'], columns='cell_id', values='value').reindex(columns=keys)
            assert np.isfinite(series.to_numpy()).all()
            clim = []
            for year in YEARS:
                a = np.array([series.loc[yr, mo].to_numpy() for (yr, mo) in [(year - 1, 10), (year - 1, 11), (year - 1, 12)] + [(year, mo) for mo in range(1, 7)]])
                clim.append(a @ W.T)
            values[v] = np.asarray(clim)
    np.savez_compressed(path, **values)
    return (family, model, 'complete')

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    numba.set_num_threads(4)
    start = time.perf_counter()
    for d in ['observed_annual_cache', 'model_annual_cache', 'source_data', 'figures', 'code']:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
    fallback = []
    for year in YEARS:
        for (yr, mo) in [(year - 1, 10), (year - 1, 11), (year - 1, 12)] + [(year, m) for m in range(1, 7)]:
            for variable in ['tmean', 'ppt', 'srad']:
                p = CLIMATE / f'TerraClimate_{variable}_{yr}_M{mo:02d}_aligned_mask.tif'
                if not p.exists():
                    source = Path('\\\\DiskStation\\data\\TerraClimate') / variable / f'TerraClimate_{variable}_{yr}.nc'
                    dest = NATIVE_CACHE / source.name
                    if not dest.exists():
                        shutil.copy2(source, dest)
                    fallback.append(dict(analysis_year=int(year), variable=variable, year=int(yr), month=mo, source=str(source), local_original_cache=str(dest), resampling='bilinear onto exact reference grid; native scale/offset retained'))
    save(OUT / 'original_climate_month_recovery.json', fallback)
    q = pixels()
    models = list(pd.read_csv(MODELROOT / 'final_all28_model_registry.csv')[['family', 'model']].itertuples(index=False, name=None))
    with ProcessPoolExecutor(max_workers=8) as pool:
        fs = [pool.submit(observed_year, int(y)) for y in YEARS] + [pool.submit(model_series, *m) for m in models]
        for (i, f) in enumerate(as_completed(fs), 1):
            print('CITY_ANNUAL', i, '/', len(fs), *f.result(), 'seconds', round(time.perf_counter() - start, 1), flush=True)
    obs = [dict(np.load(OUT / 'observed_annual_cache' / f'{y}.npz')) for y in YEARS]
    yo = np.array([v['SOS'] for v in obs])
    tc = np.array([v['tmean'] for v in obs])[:, :, None, :]
    pc = np.array([v['ppt'] for v in obs])[:, :, None, :]
    rc = np.array([v['srad'] for v in obs])[:, :, None, :]
    (lag, score, mean, observed_predictors) = optimized(yo, tc, pc, rc)
    datasets = []
    shared = np.isfinite(yo) & np.all(np.isfinite(np.stack(observed_predictors)), axis=0)
    for (family, model) in models:
        z = np.load(OUT / 'model_annual_cache' / f'{model}.npz')
        ym = z['SOS']
        if family == 'CMIP6':
            pred = common_predictors(lag, mean, *[z[k][:, :, None, :].astype(np.float32) for k in ['tas', 'pr', 'rsds']])
        else:
            pred = observed_predictors
        shared &= np.isfinite(ym) & np.all(np.isfinite(np.stack(pred)), axis=0)
        datasets.append((family, model, ym, pred))
    records = []
    obstable = []
    for (i, cid) in enumerate(q['cities']):
        valid = shared[:, i]
        o = regress(np.where(valid, yo[:, i], np.nan), *[p[:, i] for p in observed_predictors])
        obstable.append(dict(city_id=int(cid), OPL=lag[i], OPL_score=score[i], mean_SOS=mean[i], ST_observed=o['ST'], P_observed=o['Pval_T'], SE_observed=o['SE_T'], n_common_years=o['n_years']))
        for (family, model, ym, pred) in datasets:
            m = regress(np.where(valid, ym[:, i], np.nan), *[p[:, i] for p in pred])
            records.append(dict(city_id=int(cid), family=family, model=model, ST_observed=o['ST'], P_observed=o['Pval_T'], SE_observed=o['SE_T'], ST_model=m['ST'], P_model=m['Pval_T'], SE_model=m['SE_T'], n_common_years=m['n_years'], OPL=lag[i], mean_observed_SOS=mean[i]))
    allrows = pd.DataFrame(records)
    validcities = set(pd.DataFrame(obstable).dropna(subset=['ST_observed']).city_id)
    for (_, g) in allrows.groupby('model'):
        validcities &= set(g.dropna(subset=['ST_model']).city_id)
    allrows = allrows[allrows.city_id.isin(validcities)]
    coords = pd.read_csv(CITY)[['city_id', 'lon_gpkg', 'lat_gpkg', 'city_area_km2']]
    allrows = allrows.merge(coords, on='city_id', validate='many_to_one')
    observed = pd.DataFrame(obstable).merge(coords, on='city_id', validate='one_to_one')
    observed['all28_model_comparable'] = observed.city_id.isin(validcities)
    observed.to_csv(OUT / 'source_data/city_observed_annual_OLS.csv', index=False, encoding='utf-8-sig')
    allrows.to_csv(OUT / 'source_data/all28_city_annual_OLS_paired.csv', index=False, encoding='utf-8-sig')
    masks = pd.DataFrame([dict(city_id=int(cid), year=int(y), valid_all_models=bool(shared[j, i])) for (i, cid) in enumerate(q['cities']) for (j, y) in enumerate(YEARS)])
    masks.to_csv(OUT / 'source_data/city_common_valid_years.csv', index=False, encoding='utf-8-sig')
    metrics = []
    from resample_model_ST_Innovation_20261012 import metric
    for (model, g) in allrows.groupby('model'):
        metrics.append(dict(model=model, family=g.family.iloc[0], n_significant_model=int((g.P_model < 0.05).sum()), **metric(g.ST_observed.to_numpy(), g.ST_model.to_numpy())))
    ensemble = []
    escores = []
    for (family, g) in allrows.groupby('family'):
        e = g.groupby('city_id').agg(ST_model=('ST_model', 'mean'), ST_observed=('ST_observed', 'first'), P_observed=('P_observed', 'first'), n_common_years=('n_common_years', 'first'), n_models=('model', 'nunique'), model_significant_fraction=('P_model', lambda s: float((s < 0.05).mean()))).reset_index().merge(coords, on='city_id')
        e['family'] = family
        ensemble.append(e)
        escores.append(dict(family=family, **metric(e.ST_observed.to_numpy(), e.ST_model.to_numpy())))
    pd.concat(ensemble).to_csv(OUT / 'source_data/ensemble_city_annual_OLS.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(OUT / 'source_data/all28_city_annual_OLS_performance.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(escores).to_csv(OUT / 'source_data/ensemble_city_annual_OLS_performance.csv', index=False, encoding='utf-8-sig')
    summary = dict(n_city_annual_series=len(q['cities']), n_all28_comparable_cities=len(validcities), n_models=28, n_observed_significant=int(((observed.P_observed < 0.05) & observed.all28_model_comparable).sum()), recovered_climate_months=fallback, ensemble_metrics=escores)
    save(OUT / 'summary.json', summary)
    print('CITY_OLS_COMPLETE', json.dumps(summary, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
