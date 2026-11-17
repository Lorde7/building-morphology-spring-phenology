from pathlib import Path
import os, json
PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT = PACKAGE / 'outputs' / 'experiment_figure1'
CONFIG = json.loads(Path(os.environ['MORPH_INPUT_CONFIG']).read_text(encoding='utf8')) if os.environ.get('MORPH_INPUT_CONFIG') else {}

def input_path(key, relative):
    return Path(CONFIG[key]) if key in CONFIG else PACKAGE / 'data' / relative
                                                                                
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '1'
import sys, json, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import rankdata, t
from scipy.linalg import qr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
ROOT = PACKAGE
BASE = OUTPUT
OUT = OUTPUT
CITY = input_path('figure1_city_metadata', 'experiment_figure1/city_metadata.csv')
NPZ = input_path('figure1_zonal_npz', 'experiment_figure1/zonal_pixels.npz')
OLS = input_path('figure1_ols_raster', 'experiment_figure1/ST_OLS.tif')
OPL = input_path('figure1_opl_raster', 'experiment_figure1/OPL.tif')
SAMPLE = input_path('figure1_radiation_pixels', 'experiment_figure1/radiation_pixels.csv')
COVARS = ['background_DSR_monthly_mean_Wm2', 'Preseason_sos_ws', 'Preseason_sos_ppt', 'Preseason_sos_vpd', 'ESA_tree_grass_pct', 'DEM']
ALPHA = 0.05
BLOCK_METRES = 2000.0
CATEGORIES = ['negative_significant', 'negative_nonsignificant', 'positive_nonsignificant', 'positive_significant']

def jsave(path, contents):
    path.write_text(json.dumps(contents, ensure_ascii=False, indent=2), encoding='utf-8')

def read_points_by_block(path, lon, lat):
    answer = np.full(len(lon), np.nan)
    with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024):
        with rasterio.open(path) as src:
            (row, col) = rasterio.transform.rowcol(src.transform, lon, lat)
            (row, col) = (np.asarray(row), np.asarray(col))
            valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
            (bh, bw) = src.block_shapes[0]
            nc = (src.width + bw - 1) // bw
            idx = np.flatnonzero(valid)
            block = row[idx] // bh * nc + col[idx] // bw
            order = np.argsort(block, kind='stable')
            (idx, block) = (idx[order], block[order])
            cuts = np.r_[0, np.flatnonzero(np.diff(block)) + 1, len(idx)]
            for (start, stop) in zip(cuts[:-1], cuts[1:]):
                take = idx[start:stop]
                (br, bc) = divmod(int(block[start]), nc)
                (y0, x0) = (br * bh, bc * bw)
                window = Window(x0, y0, min(bw, src.width - x0), min(bh, src.height - y0))
                tile = src.read(1, window=window)
                vals = tile[row[take] - y0, col[take] - x0].astype(float)
                if src.nodata is not None:
                    vals[vals == src.nodata] = np.nan
                answer[take] = vals
    return answer

def assemble_input():
    cities = pd.read_csv(CITY)
    z = np.load(NPZ, allow_pickle=True)
    city = z['city_id']
    idx = np.flatnonzero((z['zone_type'] == 0) & np.isin(city, cities.city_id.to_numpy()))
    d = pd.DataFrame(dict(source_index=idx, city_id=city[idx].astype(int), lon=z['lon'][idx], lat=z['lat'][idx], pixel_BMSI_A=z['Building_BMSI_A_equal_tertile'][idx]))
    for col in ['Preseason_sos_ws', 'Preseason_sos_ppt', 'Preseason_sos_vpd', 'ESA_tree_grass_pct', 'DEM']:
        d[col] = z[col][idx]
    d = d.replace([np.inf, -np.inf, -9999], np.nan)
    print('ALL_URBAN_CACHE', len(d), 'READ_OLS', flush=True)
    d['ST_OLS_days_per_C'] = read_points_by_block(OLS, d.lon.to_numpy(), d.lat.to_numpy())
    print('READ_OPL', flush=True)
    d['OPL_months'] = read_points_by_block(OPL, d.lon.to_numpy(), d.lat.to_numpy())
    valid_opl = d.OPL_months.between(1, 9)
    d.loc[valid_opl, 'OPL_months'] = d.loc[valid_opl, 'OPL_months'].round()
    d.loc[~valid_opl, 'OPL_months'] = np.nan
    d['background_DSR_monthly_mean_Wm2'] = z['Preseason_sos_srad'][idx] / d.OPL_months
    d = d.replace([np.inf, -np.inf, -9999], np.nan)
    sample = pd.read_csv(SAMPLE).set_index('source_index')
    check = d.set_index('source_index').reindex(sample.index)
    for column in ['pixel_BMSI_A', 'ST_OLS_days_per_C', 'OPL_months', 'background_DSR_monthly_mean_Wm2']:
        assert np.allclose(check[column], sample[column], equal_nan=True, atol=2e-05, rtol=1e-07), column
    assert d.source_index.is_unique
    d = d[d.ST_OLS_days_per_C.notna()].copy()
    np.savez_compressed(OUT / 'all_valid_city_pixel_input.npz', **{col: d[col].to_numpy() for col in d})
    counts = d.groupby('city_id').size()
    expected = cities.set_index('city_id').ST_OLS_valid_pixel_count
    count_check = pd.DataFrame({'cached_pixels': counts, 'city_summary_raster_pixels': expected})
    count_check.to_csv(OUT / 'pixel_support_count_check.csv', encoding='utf-8-sig')
    print('VALID_OLS_PIXELS', len(d), 'CITIES', d.city_id.nunique(), flush=True)
    return (cities, d)

def fit_city(item):
    (cid, g) = item
    n_source = len(g)
    columns = ['pixel_BMSI_A', 'ST_OLS_days_per_C'] + COVARS
    q = g.dropna(subset=columns).copy()
    row = dict(city_id=int(cid), n_source=n_source, n_complete=len(q), status='not_estimable', partial_rho=np.nan, P=np.nan, P_iid_nominal=np.nan, n_spatial_blocks=0, effective_controls=0, category='not_estimable')
    if len(q) < 10:
        row['reason'] = 'fewer_than_10_complete_pixels'
        return row
    ranked = np.column_stack([rankdata(q[c].to_numpy(), method='average') for c in columns])
    (x, y) = (ranked[:, 0], ranked[:, 1])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        row['reason'] = 'no_BMSI_or_ST_variation'
        return row
    cov = ranked[:, 2:]
    sd = cov.std(axis=0)
    cov = cov[:, sd > 0]
    cov = (cov - cov.mean(axis=0)) / cov.std(axis=0) if cov.shape[1] else cov
    if cov.shape[1]:
        (_, r, pivot) = qr(cov, mode='economic', pivoting=True)
        k = int(np.sum(np.abs(np.diag(r)) > 1e-10 * np.abs(r[0, 0])))
        cov = cov[:, pivot[:k]]
    else:
        k = 0
    z = np.column_stack([np.ones(len(q)), cov])
    rx = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    ry = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    dof = len(q) - k - 2
    if dof < 5 or np.linalg.norm(rx) < 1e-08 or np.linalg.norm(ry) < 1e-08:
        row['reason'] = 'insufficient_residual_variation_or_df'
        return row
    rho = float(np.clip(np.dot(rx, ry) / np.linalg.norm(rx) / np.linalg.norm(ry), -1, 1))
    row['P_iid_nominal'] = float(2 * t.sf(abs(rho * np.sqrt(dof / max(1e-14, 1 - rho * rho))), dof))
    (lon, lat) = (q.lon.to_numpy(), q.lat.to_numpy())
    mx = (lon - lon.mean()) * 111320 * np.cos(np.deg2rad(lat.mean()))
    my = (lat - lat.mean()) * 110574
    labels = np.c_[np.floor(mx / BLOCK_METRES), np.floor(my / BLOCK_METRES)]
    (_, blocks) = np.unique(labels, axis=0, return_inverse=True)
    nb = int(blocks.max() + 1)
    row.update(partial_rho=rho, n_spatial_blocks=nb, effective_controls=k)
    if nb < 10:
        row['reason'] = 'fewer_than_10_spatial_blocks'
        return row
    xz = (x - x.mean()) / x.std()
    yz = (y - y.mean()) / y.std()
    design = np.column_stack([z, xz])
    fit = sm.OLS(yz, design).fit(cov_type='cluster', cov_kwds={'groups': blocks, 'use_correction': True}, use_t=True)
    p = float(fit.pvalues[-1])
    if not np.isfinite(p):
        row['reason'] = 'nonfinite_spatial_P'
        return row
    assert np.sign(fit.params[-1]) == np.sign(rho)
    direction = 'negative' if rho < 0 else 'positive'
    row.update(P=p, status='estimated', reason='', category=direction + ('_significant' if p < ALPHA else '_nonsignificant'))
    return row

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    (cities, d) = assemble_input()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(fit_city, d.groupby('city_id', sort=True)))
    results = cities[['city_id', 'lon_gpkg', 'lat_gpkg']].merge(pd.DataFrame(rows), on='city_id', how='left')
    results['status'] = results.status.fillna('not_estimable')
    results['category'] = results.category.fillna('not_estimable')
    results['reason'] = results.reason.fillna('no_valid_cached_urban_pixels')
    ok = results.status.eq('estimated')
    results.loc[ok, 'q_BH'] = multipletests(results.loc[ok, 'P'], method='fdr_bh')[1]
    results.to_csv(OUT / 'city_partial_correlations.csv', index=False, encoding='utf-8-sig')
    counts = results.loc[ok, 'category'].value_counts().reindex(CATEGORIES, fill_value=0)
    summary = pd.DataFrame({'category': CATEGORIES, 'n_cities': counts.to_numpy(), 'percent': counts.to_numpy() / ok.sum() * 100, 'denominator_n': int(ok.sum())})
    summary.to_csv(OUT / 'category_summary.csv', index=False, encoding='utf-8-sig')
    reasons = results.loc[~ok, 'reason'].value_counts().to_dict()
    print(summary.to_string(index=False), flush=True)
    print('NOT_ESTIMABLE', reasons, flush=True)
    print('CITY_PIXEL_PARTIAL_COMPLETE', OUT, flush=True)
if __name__ == '__main__':
    main()
