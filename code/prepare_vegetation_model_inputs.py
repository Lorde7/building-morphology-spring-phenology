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
import sys, json, calendar, math, time, shutil
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
from rasterio.warp import reproject
from affine import Affine
ROOT = Path(configured_path('prepare_TRENDY_historical_ST_20261010.path1'))
OUT = ROOT / 'common_2degree_benchmark'
SCALE = 2.0
DEST_TRANSFORM = Affine(SCALE, 0, -180, 0, -SCALE, 90)
GPP = Path('\\\\DiskStation\\data\\trendy-v11_new\\TRENDY_Models')
SOS = Path('\\\\DiskStation\\data\\DataSet_Modis\\MCD12Q2.061\\TIF\\Greenup_Cycle1_ReProj')
MODELS = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'DLEM', 'IBIS', 'ISAM', 'JSBACH', 'JULES', 'LPJ-GUESS', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'VISIT', 'VISIT-NIES', 'YIBs']
CITY = Path(configured_path('prepare_TRENDY_historical_ST_20261010.path2'))
REF = Path(configured_path('prepare_TRENDY_historical_ST_20261010.path3'))

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')

def gpp_job(model, grid):
    cache = OUT / 'monthly_GPP' / f'{model}.csv'
    if cache.exists():
        q = pd.read_csv(cache, dtype={'cell_id': str})
        assert len(q) == len(grid) * 240
        return ('gpp', model, len(q))
    rows = []
    meta = []
    rr = grid.model_row.to_numpy(int)
    cc = grid.model_col.to_numpy(int)
    for year in range(2001, 2021):
        for month in range(1, 13):
            p = GPP / model / 'gpp' / f'gpp_{year}-{month:02d}.tif'
            with MemoryFile(p.read_bytes()) as memory, memory.open() as src:
                assert src.crs.to_epsg() == 4326
                assert max(abs(src.transform.a), abs(src.transform.e)) <= SCALE
                a = src.read(1, masked=True).astype(float).filled(np.nan)
                a[a < 0] = np.nan
                lat = src.transform.f + (np.arange(src.height) + 0.5) * src.transform.e
                weight = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape).copy()
                weight[~np.isfinite(a)] = np.nan
                num = np.full((int(round(180 / SCALE)), int(round(360 / SCALE))), np.nan)
                den = num.copy()
                for (source, dest) in [(a * weight, num), (weight, den)]:
                    reproject(source, dest, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan, dst_transform=DEST_TRANSFORM, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.average)
                aggregate = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
                values = aggregate[rr, cc]
                values[values < 0] = np.nan
                if year == 2001 and month == 1:
                    meta.append(dict(path=str(p), grid=[src.width, src.height], transform=list(src.transform), tags=src.tags(), description=src.descriptions, unit='Local historical GPP raster has no unit attribute; relative seasonal threshold is invariant to a constant positive scale', original_native_resolution='Not inferred: existing geographic converted grids range 0.5 to 1.875 degrees; benchmark conservatively aggregates to 2 degrees'))
            for (cell, val) in zip(grid.cell_id, values):
                rows.append(dict(model=model, cell_id=cell, year=year, month=month, GPP=val))
    pd.DataFrame(rows).to_csv(cache, index=False, encoding='utf-8-sig')
    save(cache.with_suffix('.json'), meta)
    return ('gpp', model, len(rows))

def sos_job(year, grid):
    cache = OUT / 'satellite_SOS' / f'SOS_{year}.csv'
    if cache.exists():
        return ('sos', year, len(pd.read_csv(cache)))
    rows = []
    p = SOS / f'Greenup_Cycle1_ReProj_{year}.tif'
    with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024), MemoryFile(p.read_bytes()) as memory, memory.open() as src:
        assert src.crs.to_epsg() == 4326
        for rec in grid.itertuples():
            (left, top) = (-180 + SCALE * rec.model_col, 90 - SCALE * rec.model_row)
            win = from_bounds(left, top - SCALE, left + SCALE, top, src.transform)
            c0 = int(math.ceil(win.col_off - 0.5))
            c1 = int(math.ceil(win.col_off + win.width - 0.5))
            r0 = int(math.ceil(win.row_off - 0.5))
            r1 = int(math.ceil(win.row_off + win.height - 0.5))
            c0 = max(0, c0)
            r0 = max(0, r0)
            c1 = min(src.width, c1)
            r1 = min(src.height, r1)
            if c1 <= c0 or r1 <= r0:
                rows.append(dict(cell_id=rec.cell_id, year=year, SOS=np.nan, n_valid_500m_pixels=0))
                continue
            raw = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), masked=True).astype(float).filled(np.nan)
            good = np.isfinite(raw) & (raw != 32767)
            dates = np.datetime64('1970-01-01') + np.where(good, raw, 0).astype('timedelta64[D]')
            doy = (dates - dates.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) + 1
            valid = good & (doy >= 1) & (doy < 190)
            latitude = src.transform.f + (np.arange(r0, r1) + 0.5) * src.transform.e
            weight = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], raw.shape)
            value = float(np.sum(doy[valid] * weight[valid]) / np.sum(weight[valid])) if valid.any() else np.nan
            rows.append(dict(cell_id=rec.cell_id, year=year, SOS=value, n_valid_500m_pixels=int(valid.sum())))
    pd.DataFrame(rows).to_csv(cache, index=False, encoding='utf-8-sig')
    return ('sos', year, len(rows))

def climate_job(var, year, grid):
    cache = OUT / 'monthly_climate' / f'{var}_{year}.csv'
    if cache.exists():
        return ('climate', f'{var}_{year}', len(pd.read_csv(cache)))
    p = Path('\\\\DiskStation\\data\\TerraClimate') / var / f'TerraClimate_{var}_{year}.nc'
    if not p.exists():
        p = Path('\\\\DiskStation\\data\\Dataset_TerraClimate') / ('TerraClimate_' + var) / f'TerraClimate_{var}_{year}.nc'
    values = np.full((12, len(grid)), np.nan)
    stage = Path(configured_path('prepare_TRENDY_historical_ST_20261010.path4'))
    stage.mkdir(exist_ok=True)
    local = stage / f'worker_{os.getpid()}.nc'
    shutil.copyfile(p, local)
    with rasterio.Env(GDAL_CACHEMAX=96 * 1024 * 1024), rasterio.open(local) as src:
        (rr, cc) = rasterio.transform.rowcol(src.transform, grid.longitude, grid.latitude)
        rr = np.asarray(rr)
        cc = np.asarray(cc)
        (bh, bw) = src.block_shapes[0]
        blocks = {}
        for i in range(len(grid)):
            blocks.setdefault((rr[i] // bh, cc[i] // bw), []).append(i)
        for ((br, bc), indices) in blocks.items():
            win = Window(bc * bw, br * bh, min(bw, src.width - bc * bw), min(bh, src.height - br * bh))
            a = src.read(window=win, masked=True).astype(float).filled(np.nan)
            ix = np.array(indices)
            values[:, ix] = a[:, rr[ix] - br * bh, cc[ix] - bc * bw]
        for m in range(12):
            values[m] = values[m] * src.scales[m] + src.offsets[m]
        meta = dict(source=str(p), scale=list(src.scales), offset=list(src.offsets), spatial_rule='Native TerraClimate cell at the common model-cell centre; no city or source validity mask', climate_comparison='Same monthly climate used for satellite and every reanalysis-forced TRENDY dataset at the same cell')
    if local.resolve().parent == stage.resolve():
        local.unlink()
    rows = []
    for m in range(12):
        for (cell, value) in zip(grid.cell_id, values[m]):
            rows.append(dict(cell_id=cell, year=year, month=m + 1, variable=var, value=value))
    pd.DataFrame(rows).to_csv(cache, index=False, encoding='utf-8-sig')
    save(cache.with_suffix('.json'), meta)
    return ('climate', f'{var}_{year}', len(rows))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.perf_counter()
    for name in ['monthly_GPP', 'monthly_climate', 'satellite_SOS', 'code']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    city = pd.read_csv(CITY, dtype={'city_id': str})
    assert len(city) == 638
    city['model_row'] = np.floor((90 - city.lat_gpkg) / SCALE).astype(int)
    city['model_col'] = np.floor((city.lon_gpkg + 180) / SCALE).astype(int)
    city['cell_id'] = city.model_row.astype(str) + ':' + city.model_col.astype(str)
    grid = city[['cell_id', 'model_row', 'model_col']].drop_duplicates('cell_id').sort_values(['model_row', 'model_col']).reset_index(drop=True)
    grid['longitude'] = -180 + SCALE * (grid.model_col + 0.5)
    grid['latitude'] = 90 - SCALE * (grid.model_row + 0.5)
    grid.to_csv(OUT / 'common_2degree_cells_containing_cities.csv', index=False, encoding='utf-8-sig')
    city.to_csv(OUT / 'original_638_cities_model_grid_mapping.csv', index=False, encoding='utf-8-sig')
    jobs = [('gpp', model) for model in MODELS] + [('sos', year) for year in range(2001, 2021)] + [('climate', (var, year)) for var in ['tmin', 'tmax', 'ppt', 'srad'] for year in range(2000, 2021)]
    print('MODEL_PREP', len(grid), 'unique 2-degree cells;', len(jobs), 'jobs; workers=12', flush=True)
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = []
        for (kind, item) in jobs:
            futures.append(pool.submit(gpp_job, item, grid) if kind == 'gpp' else pool.submit(sos_job, item, grid) if kind == 'sos' else pool.submit(climate_job, *item, grid))
        done = []
        for (i, f) in enumerate(as_completed(futures), 1):
            result = f.result()
            done.append(result)
            print('MODEL_DATA', i, '/', len(jobs), *result, 'seconds', round(time.perf_counter() - started, 1), flush=True)
if __name__ == '__main__':
    main()
