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
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[k] = '1'
import sys, json, time, shutil
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
from scipy.stats import pearsonr
from model_paths import ROOT, OUT as MODELROOT
OUT = ROOT / 'Innovation_resampling_20261012'
RESULT = MODELROOT / 'ST_comparison_all_models'
CITY = Path(configured_path('resample_model_ST_Innovation_20261012.path1'))
PIX = Path(configured_path('resample_model_ST_Innovation_20261012.path2'))
ST = Path(configured_path('resample_model_ST_Innovation_20261012.path3'))

def writejson(p, o):
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding='utf-8')

def coarse(q, key):
    arr = np.full((60, 120), np.nan, dtype=np.float32)
    rc = np.array([list(map(int, x.split(':'))) for x in q.cell_id])
    arr[rc[:, 0], rc[:, 1]] = q[key].to_numpy(np.float32)
    return arr

def interp(a, pix):
    x = (pix['lon_exact'] + 180) / 3 - 0.5
    y = (90 - pix['lat_exact']) / 3 - 0.5
    c = np.floor(x).astype(int)
    r = np.floor(y).astype(int)
    dx = x - c
    dy = y - r
    num = np.zeros(len(x))
    den = np.zeros(len(x))
    for (dr, dc, w) in [(0, 0, (1 - dx) * (1 - dy)), (0, 1, dx * (1 - dy)), (1, 0, (1 - dx) * dy), (1, 1, dx * dy)]:
        v = a[np.clip(r + dr, 0, 59), (c + dc) % 120]
        ok = np.isfinite(v)
        num += np.where(ok, v * w, 0)
        den += np.where(ok, w, 0)
    result = np.divide(num, den, out=np.full(len(x), np.nan), where=den > 0)
    return result.astype(np.float32)

def weighted_city(values, pix):
    ids = pix['city_index']
    w = pix['pixel_area_weight']
    n = int(ids.max() + 1)
    ok = np.isfinite(values)
    num = np.bincount(ids, weights=np.where(ok, values * w, 0), minlength=n)
    den = np.bincount(ids, weights=np.where(ok, w, 0), minlength=n)
    return np.divide(num, den, out=np.full(n, np.nan), where=den > 0)

def metric(obs, pred):
    ok = np.isfinite(obs) & np.isfinite(pred)
    a = obs[ok]
    b = pred[ok]
    e = b - a
    return dict(n_cities=int(ok.sum()), Pearson_r=float(pearsonr(a, b)[0]), RMSE_ST=float(np.sqrt(np.mean(e * e))), MAE_ST=float(np.mean(abs(e))), mean_bias_ST=float(np.mean(e)), predictive_R2=float(1 - np.sum(e * e) / np.sum((a - a.mean()) ** 2)), same_sign_fraction=float(np.mean(np.sign(a) == np.sign(b))))

def get_observed():
    dest = OUT / 'urban_pixel_observed_and_grid.npz'
    if dest.exists():
        return {k: v for (k, v) in np.load(dest).items()}
    z = np.load(PIX, allow_pickle=True)
    targets = pd.read_csv(CITY).city_id.to_numpy(int)
    select = (z['zone_type'] == 0) & np.isin(z['city_id'], targets)
    lo = z['lon'][select].astype(float)
    la = z['lat'][select].astype(float)
    cid = z['city_id'][select].astype(int)
    del z
    with rasterio.open(ST) as src:
        (rr, cc) = rasterio.transform.rowcol(src.transform, lo, la)
        rr = np.asarray(rr, int)
        cc = np.asarray(cc, int)
        assert ((rr >= 0) & (rr < src.height) & (cc >= 0) & (cc < src.width)).all()
        (bh, bw) = src.block_shapes[0]
        bid = rr // bh * int(np.ceil(src.width / bw)) + cc // bw
        sort = np.argsort(bid)
        splits = np.r_[0, np.flatnonzero(np.diff(bid[sort])) + 1, len(sort)]
        vv = np.full((2, len(lo)), np.nan, dtype=np.float32)
        for j in range(len(splits) - 1):
            ix = sort[splits[j]:splits[j + 1]]
            r0 = rr[ix[0]] // bh * bh
            c0 = cc[ix[0]] // bw * bw
            a = src.read([1, 7], window=Window(c0, r0, min(bw, src.width - c0), min(bh, src.height - r0)), masked=True).astype(np.float32).filled(np.nan)
            vv[:, ix] = a[:, rr[ix] - r0, cc[ix] - c0]
        tr = src.transform
        profile = src.profile.copy()
    ok = np.isfinite(vv[0])
    lo = lo[ok]
    la = la[ok]
    cid = cid[ok]
    rr = rr[ok]
    cc = cc[ok]
    vv = vv[:, ok]
    lon_exact = tr.c + (cc + 0.5) * tr.a
    lat_exact = tr.f + (rr + 0.5) * tr.e
    assert np.max(abs(lon_exact - lo)) < 2e-05 and np.max(abs(lat_exact - la)) < 2e-05
    (cities, index) = np.unique(cid, return_inverse=True)
    pix = dict(lon_exact=lon_exact, lat_exact=lat_exact, city_id=cid, city_index=index.astype(np.int32), cities=cities, row=rr.astype(np.int32), col=cc.astype(np.int32), ST_observed=vv[0], P_observed=vv[1], coarse_row=np.floor((90 - lat_exact) / 3).astype(np.int16), coarse_col=np.floor((lon_exact + 180) / 3).astype(np.int16), pixel_area_weight=np.cos(np.deg2rad(lat_exact)))
    np.savez_compressed(dest, **pix)
    profile.update(count=2, dtype='float32', nodata=-9999.0, compress='lzw', predictor=2, tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER', SPARSE_OK='TRUE')
    writejson(OUT / 'target_grid.json', dict(reference=str(ST), width=profile['width'], height=profile['height'], transform=list(profile['transform']), crs=str(profile['crs']), n_original_urban_pixels=int(select.sum()), n_valid_observed_pixels=len(lo), n_valid_observed_cities=len(cities), zone_type=0, full_urban_pixels=True, new_sampling=False, observed_ST_P_selection=False))
    print('OBSERVED', len(lo), 'urban pixels;', len(cities), 'cities;', len(splits) - 1, 'blocks', flush=True)
    return pix

def one(family, model):
    pix = {k: v for (k, v) in np.load(OUT / 'urban_pixel_observed_and_grid.npz').items()}
    p = pd.read_csv(RESULT / 'all_models_common_years_shared_window_ST.csv', dtype={'cell_id': str})
    q = p[p.model.eq(model)]
    a = coarse(q, 'ST_model_matched')
    pp = coarse(q, 'P_model')
    nearest = a[pix['coarse_row'], pix['coarse_col']]
    pn = pp[pix['coarse_row'], pix['coarse_col']]
    bilinear = interp(a, pix)
    bilinear[~np.isfinite(nearest)] = np.nan
    np.savez_compressed(OUT / 'resampled_pixel_models' / f'{model}.npz', ST_bilinear=bilinear, ST_nearest=nearest, P_source_nearest=pn)
    return (family, model, int(np.isfinite(bilinear).sum()))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    start = time.perf_counter()
    for folder in ['code', 'resampled_pixel_models', 'source_data', 'figures', 'rasters']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    pix = get_observed()
    registry = pd.read_csv(MODELROOT / 'final_all28_model_registry.csv')
    models = list(zip(registry.family, registry.model))
    errors = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        fs = {pool.submit(one, *m): m for m in models}
        for f in as_completed(fs):
            try:
                print('RESAMPLED', *f.result(), flush=True)
            except Exception as e:
                errors.append(dict(model=fs[f], error=str(e)))
    if errors:
        raise RuntimeError(errors)
    common = np.ones(len(pix['city_id']), bool)
    for (_, model) in models:
        z = np.load(OUT / 'resampled_pixel_models' / f'{model}.npz')
        common &= np.isfinite(z['ST_bilinear']) & np.isfinite(z['ST_nearest'])
    np.save(OUT / 'common_urban_pixel_mask.npy', common)
    sub = {k: v[common] if np.ndim(v) == 1 and len(v) == len(common) else v for (k, v) in pix.items()}
    (cities, index) = np.unique(sub['city_id'], return_inverse=True)
    sub['cities'] = cities
    sub['city_index'] = index.astype(np.int32)
    obs = weighted_city(sub['ST_observed'], sub)
    annual_city = pd.read_csv(CITY).set_index('city_id')
    base = pd.DataFrame(dict(city_id=cities, ST_observed_all_common_urban_pixels=obs, n_common_urban_pixels=np.bincount(index), original_full_city_ST=annual_city.loc[cities, 'ST_OLS_days_per_C'].to_numpy()))
    base.to_csv(OUT / 'source_data/city_observed_common_support.csv', index=False, encoding='utf-8-sig')
    table = []
    metrics = []
    details = []
    classrows = []
    ensembles = {'TRENDY': np.zeros(len(cities)), 'CMIP6': np.zeros(len(cities))}
    ensemble_pixel = {'TRENDY': np.zeros(common.sum()), 'CMIP6': np.zeros(common.sum())}
    nfamily = registry.family.value_counts().to_dict()
    obs_neg = weighted_city((sub['ST_observed'] < 0).astype(float), sub)
    obs_sig = weighted_city((sub['P_observed'] < 0.05).astype(float), sub)
    classrows.append(dict(family='Observed', model='Observed', negative_fraction_city_equal=float(obs_neg.mean()), significant_fraction_city_equal=float(obs_sig.mean()), n_pixels=int(common.sum()), n_cities=len(cities)))
    for (family, model) in models:
        z = np.load(OUT / 'resampled_pixel_models' / f'{model}.npz')
        bl = z['ST_bilinear'][common]
        nn = z['ST_nearest'][common]
        pn = z['P_source_nearest'][common]
        val = weighted_city(bl, sub)
        valnn = weighted_city(nn, sub)
        metrics.append(dict(family=family, model=model, resampling='bilinear', **metric(obs, val)))
        metrics.append(dict(family=family, model=model, resampling='nearest', **metric(obs, valnn)))
        ensembles[family] += val / nfamily[family]
        ensemble_pixel[family] += bl / nfamily[family]
        table.append(pd.DataFrame(dict(family=family, model=model, city_id=cities, ST_observed=obs, ST_model_bilinear=val, ST_model_nearest=valnn, bias_bilinear=val - obs, n_common_urban_pixels=np.bincount(index))))
        negative = weighted_city((nn < 0).astype(float), sub)
        sig = weighted_city((pn < 0.05).astype(float), sub)
        classrows.append(dict(family=family, model=model, negative_fraction_city_equal=float(negative.mean()), significant_fraction_city_equal=float(sig.mean()), n_pixels=int(common.sum()), n_cities=len(cities)))
        agree = np.sign(nn) == np.sign(sub['ST_observed'])
        so = sub['P_observed'] < 0.05
        sm = pn < 0.05
        categories = {'both_significant': agree & so & sm, 'observed_only_significant': agree & so & ~sm, 'model_only_significant': agree & ~so & sm, 'neither_significant': agree & ~so & ~sm, 'opposite_direction': ~agree}
        row = dict(family=family, model=model)
        for (name, mask) in categories.items():
            row[name] = float(weighted_city(mask.astype(float), sub).mean())
        assert abs(sum((row[n] for n in categories)) - 1) < 1e-08
        details.append(row)
        print('CITY_COMPARE', model, round(metrics[-2]['Pearson_r'], 3), round(metrics[-2]['RMSE_ST'], 3), flush=True)
    pd.concat(table).to_csv(OUT / 'source_data/all28_city_ST_paired.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(OUT / 'source_data/all28_city_model_performance.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(classrows).to_csv(OUT / 'source_data/pixel_direction_and_source_significance.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(OUT / 'source_data/direction_agreement_by_significance.csv', index=False, encoding='utf-8-sig')
    ens = []
    scores = []
    for (fam, pred) in list(ensembles.items()) + [('AllModels', (ensembles['TRENDY'] * 15 + ensembles['CMIP6'] * 13) / 28)]:
        ens.append(pd.DataFrame(dict(family=fam, city_id=cities, ST_observed=obs, ST_model=pred)))
        scores.append(dict(family=fam, **metric(obs, pred)))
    pd.concat(ens).to_csv(OUT / 'source_data/ensemble_city_ST.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(OUT / 'source_data/ensemble_city_performance.csv', index=False, encoding='utf-8-sig')
    sourceids = sub['coarse_row'].astype(int) * 120 + sub['coarse_col']
    np.savez_compressed(OUT / 'ensemble_resampled_urban_pixels.npz', row=sub['row'], col=sub['col'], city_id=sub['city_id'], ST_observed=sub['ST_observed'], ST_TRENDY=ensemble_pixel['TRENDY'], ST_CMIP6=ensemble_pixel['CMIP6'], source_row=sub['coarse_row'], source_col=sub['coarse_col'])
    writejson(OUT / 'summary.json', dict(n_models=28, n_original_cities=638, n_common_cities=len(cities), n_common_urban_pixels=int(common.sum()), ensemble_metrics=scores))
if __name__ == '__main__':
    main()
