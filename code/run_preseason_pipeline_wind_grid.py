from __future__ import annotations
import argparse
import csv
import gc
import math
import os
import re
import time
from pathlib import Path
import numba as nb
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.stats import t as student_t
from tqdm import tqdm
CLIMATE_DIR = Path(os.environ.get('MORPH_CLIMATE_DIR', str(Path(__file__).resolve().parents[1] / 'data' / 'climate')))
OUT_ROOT = Path(os.environ.get('MORPH_ST_WORKSPACE', str(Path(__file__).resolve().parents[1] / 'outputs' / 'ST_workspace')))
PHENO_YEARS = list(range(2001, 2021))
CLIMATE_YEARS = list(range(min(PHENO_YEARS) - 1, max(PHENO_YEARS) + 1))
MONTH_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], dtype=np.int32)
OUT_NODATA = -9999.0
SOURCE_PHENO_NODATA = 32767
MAX_LAG = 6
THRESHOLD = 0.5
MIN_VALID_YEARS = 11
RIDGE_ALPHA = 1.0
RIDGE_METHOD = 'Wang2022_minmax_anomalies_MATLAB_ridge_k1'
TILE_HEIGHT = 128
TILE_WIDTH = 512
OPL_VARS = ['tmean', 'ppt', 'srad']
SUM_VARS = {'ppt', 'srad', 'def'}
SENSITIVITY_VARS = ['tmean', 'ppt', 'srad']
PHENO_TARGETS = ['sos', 'eos']
BAND_DESCRIPTIONS = ['Ridge_T', 'Ridge_P', 'Ridge_R', 'PartialCorr_T', 'PartialCorr_P', 'PartialCorr_R', 'Pval_T', 'Pval_P', 'Pval_R']

def stamp() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log_line(log_path: Path, msg: str) -> None:
    text = f'[{stamp()}] {msg}'
    print(text, flush=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(text + '\n')

def ensure_dirs() -> dict[str, Path]:
    dirs = {'phenology': OUT_ROOT / 'phenology_doy', 'opl': OUT_ROOT / 'OPL_preseason_results', 'preseason': OUT_ROOT / 'Preseason_climate_results', 'sensitivity': OUT_ROOT / 'Sensitivity_results', 'logs': OUT_ROOT / 'logs'}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

def reference_path() -> Path:
    p = CLIMATE_DIR / 'TerraClimate_ws_2001_M01_aligned_mask.tif'
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def pheno_source_path(pheno: str, year: int) -> Path:
    return CLIMATE_DIR / f'MODIS_MCD12Q2_{pheno}_{year}_aligned_wsmask.tif'

def pheno_doy_path(pheno: str, year: int, dirs: dict[str, Path]) -> Path:
    return dirs['phenology'] / f'{pheno}_{year}.tif'

def climate_path(var: str, year: int, month: int) -> Path:
    return CLIMATE_DIR / f'TerraClimate_{var}_{year}_M{month:02d}_aligned_mask.tif'

def opl_lag_path(pheno: str, dirs: dict[str, Path]) -> Path:
    if pheno == 'sos':
        return dirs['opl'] / 'OPL_sos_prevOct_preseason_lag_merged.tif'
    return dirs['opl'] / 'OPL_eos_relative_preseason_lag_merged.tif'

def opl_score_path(pheno: str, dirs: dict[str, Path]) -> Path:
    if pheno == 'sos':
        return dirs['opl'] / 'OPL_sos_prevOct_preseason_score_merged.tif'
    return dirs['opl'] / 'OPL_eos_relative_preseason_score_merged.tif'

def preseason_path(pheno: str, var: str, dirs: dict[str, Path]) -> Path:
    stat = 'sum' if var in SUM_VARS else 'mean'
    return dirs['preseason'] / f'Preseason_{pheno}_{var}_{stat}.tif'

def sensitivity_path(pheno: str, dirs: dict[str, Path]) -> Path:
    return dirs['sensitivity'] / f'ridgeAnalysis_{pheno}_9bands.tif'

def output_profile_from_ref(ref_path: Path, dtype: str, count: int, nodata: float=OUT_NODATA) -> dict:
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
    profile.update(driver='GTiff', dtype=dtype, count=count, nodata=nodata, compress='lzw', predictor=2 if dtype in {'float32', 'int16'} else None, tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
    if profile.get('predictor') is None:
        profile.pop('predictor', None)
    return profile

def iter_windows(width: int, height: int, tile_h: int=TILE_HEIGHT, tile_w: int=TILE_WIDTH):
    for row in range(0, height, tile_h):
        h = min(tile_h, height - row)
        for col in range(0, width, tile_w):
            w = min(tile_w, width - col)
            yield Window(col, row, w, h)

def read_tile(path: Path | None, win: Window) -> np.ndarray:
    shape = (int(win.height), int(win.width))
    if path is None or not path.exists():
        return np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        arr = src.read(1, window=win).astype(np.float32, copy=False)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    arr[arr < -1000] = np.nan
    return arr

def convert_modis_date_tile_to_doy(raw: np.ndarray, year: int, pheno: str, nodata: float | int | None) -> np.ndarray:
    out = np.full(raw.shape, OUT_NODATA, dtype=np.float32)
    valid = np.isfinite(raw)
    if nodata is not None:
        valid &= raw != nodata
    valid &= raw != SOURCE_PHENO_NODATA
    if not np.any(valid):
        return out
    vals = raw[valid].astype('int64')
    dates = np.datetime64('1970-01-01') + vals.astype('timedelta64[D]')
    years = dates.astype('datetime64[Y]')
    doy = (dates - years).astype('timedelta64[D]').astype(np.int32) + 1
    if pheno == 'sos':
        keep = (doy >= 1) & (doy < 190)
    elif pheno == 'eos':
        keep = (doy > 220) & (doy <= 366)
    else:
        keep = (doy >= 1) & (doy <= 366)
    idx = np.flatnonzero(valid)
    flat = out.ravel()
    flat[idx[keep]] = doy[keep].astype(np.float32)
    return out

def create_pheno_doy(dirs: dict[str, Path], overwrite: bool, log_path: Path) -> None:
    ref = reference_path()
    profile = output_profile_from_ref(ref, 'float32', 1, OUT_NODATA)
    with rasterio.open(ref) as src:
        (width, height) = (src.width, src.height)
    for pheno in PHENO_TARGETS:
        for year in PHENO_YEARS:
            src_path = pheno_source_path(pheno, year)
            out_path = pheno_doy_path(pheno, year, dirs)
            if out_path.exists() and (not overwrite):
                continue
            if not src_path.exists():
                raise FileNotFoundError(src_path)
            log_line(log_path, f'Create DOY {pheno} {year}: {out_path}')
            with rasterio.open(src_path) as src, rasterio.open(out_path, 'w', **profile) as dst:
                for (_, win) in src.block_windows(1):
                    raw = src.read(1, window=win).astype(np.float32, copy=False)
                    doy = convert_modis_date_tile_to_doy(raw, year, pheno, src.nodata)
                    dst.write(doy, 1, window=win)
                dst.update_tags(source=str(src_path), source_date_units='days since 1970-01-01', converted_to='day_of_year', valid_rule='sos: DOY < 190; eos: DOY > 220')

def read_pheno_stack(pheno: str, win: Window, dirs: dict[str, Path]) -> np.ndarray:
    arrs = []
    for year in PHENO_YEARS:
        arrs.append(read_tile(pheno_doy_path(pheno, year, dirs), win))
    return np.stack(arrs, axis=0).astype(np.float32)

def read_sos_climate_stack(var: str, win: Window) -> np.ndarray:
    year_blocks = []
    for year in PHENO_YEARS:
        months = [(year - 1, 10), (year - 1, 11), (year - 1, 12)] + [(year, m) for m in range(1, 7)]
        month_blocks = [read_tile(climate_path(var, cy, cm), win) for (cy, cm) in months]
        year_blocks.append(np.stack(month_blocks, axis=0))
    return np.stack(year_blocks, axis=0).astype(np.float32)

def read_eos_climate_stack(var: str, win: Window) -> np.ndarray:
    month_blocks = []
    for year in CLIMATE_YEARS:
        for month in range(1, 13):
            month_blocks.append(read_tile(climate_path(var, year, month), win))
    return np.stack(month_blocks, axis=0).astype(np.float32)

@nb.njit(cache=True)
def doy_to_month_numba(doy: float) -> int:
    if not math.isfinite(doy):
        return -1
    d = int(round(doy))
    if d < 1:
        d = 1
    if d > 365:
        d = 365
    if d <= 31:
        return 1
    if d <= 59:
        return 2
    if d <= 90:
        return 3
    if d <= 120:
        return 4
    if d <= 151:
        return 5
    if d <= 181:
        return 6
    if d <= 212:
        return 7
    if d <= 243:
        return 8
    if d <= 273:
        return 9
    if d <= 304:
        return 10
    if d <= 334:
        return 11
    return 12

@nb.njit(cache=True)
def detrend_columns(data: np.ndarray, n: int, p: int) -> None:
    mean_t = (n - 1) * 0.5
    denom_t = 0.0
    for i in range(n):
        dt = i - mean_t
        denom_t += dt * dt
    if denom_t <= 0:
        return
    for j in range(p):
        mean_y = 0.0
        for i in range(n):
            mean_y += data[i, j]
        mean_y /= n
        cov = 0.0
        for i in range(n):
            cov += (i - mean_t) * (data[i, j] - mean_y)
        slope = cov / denom_t
        intercept = mean_y - slope * mean_t
        for i in range(n):
            data[i, j] = data[i, j] - (intercept + slope * i)

@nb.njit(cache=True)
def partial_corr_score(data: np.ndarray, n: int, p: int) -> float:
    mean = np.zeros(p, dtype=np.float64)
    std = np.zeros(p, dtype=np.float64)
    for j in range(p):
        for i in range(n):
            mean[j] += data[i, j]
        mean[j] /= n
        for i in range(n):
            d = data[i, j] - mean[j]
            std[j] += d * d
        std[j] = math.sqrt(std[j] / n)
        if std[j] <= 0 or not math.isfinite(std[j]):
            return -1.0
    keep = np.ones(n, dtype=np.bool_)
    nk = 0
    for i in range(n):
        ok = True
        for j in range(p):
            z = abs((data[i, j] - mean[j]) / std[j])
            if z >= 3.0:
                ok = False
                break
        keep[i] = ok
        if ok:
            nk += 1
    if nk < 2:
        return -1.0
    work = np.empty((nk, p), dtype=np.float64)
    r = 0
    for i in range(n):
        if keep[i]:
            for j in range(p):
                work[r, j] = data[i, j]
            r += 1
    detrend_columns(work, nk, p)
    for j in range(p):
        m = 0.0
        for i in range(nk):
            m += work[i, j]
        m /= nk
        ss = 0.0
        for i in range(nk):
            work[i, j] -= m
            ss += work[i, j] * work[i, j]
        if ss <= 0 or not math.isfinite(ss):
            return -1.0
        scale = math.sqrt(ss)
        for i in range(nk):
            work[i, j] /= scale
    corr = np.empty((p, p), dtype=np.float64)
    for a in range(p):
        for b in range(p):
            s = 0.0
            for i in range(nk):
                s += work[i, a] * work[i, b]
            corr[a, b] = s
    for j in range(p):
        corr[j, j] += 1e-08
    inv = np.linalg.inv(corr)
    denom = inv[0, 0] * inv[1, 1]
    if denom <= 0 or not math.isfinite(denom):
        return -1.0
    pc = -inv[0, 1] / math.sqrt(denom)
    if not math.isfinite(pc):
        return -1.0
    if pc < 0:
        pc = -pc
    return pc

@nb.njit(cache=True, parallel=True)
def compute_opl_sos(pheno: np.ndarray, tmean: np.ndarray, ppt: np.ndarray, srad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (ny, h, w) = pheno.shape
    lag_out = np.full((h, w), OUT_NODATA, dtype=np.float32)
    score_out = np.full((h, w), OUT_NODATA, dtype=np.float32)
    min_n = int(math.ceil(ny * THRESHOLD))
    for rr in nb.prange(h):
        data = np.empty((ny, 4), dtype=np.float64)
        for cc in range(w):
            nph = 0
            ymean = 0.0
            for y in range(ny):
                val = pheno[y, rr, cc]
                if math.isfinite(val):
                    nph += 1
                    ymean += val
            if nph < min_n:
                continue
            ymean /= nph
            month = doy_to_month_numba(ymean)
            if month < 1:
                continue
            if month > 6:
                month = 6
            target_idx = month + 2
            best_lag = -1
            best_score = -1.0
            for lag in range(1, MAX_LAG + 1):
                start_idx = target_idx - lag + 1
                if start_idx < 0:
                    continue
                n = 0
                for y in range(ny):
                    yv = pheno[y, rr, cc]
                    if not math.isfinite(yv):
                        continue
                    vals = np.zeros(3, dtype=np.float64)
                    cnts = np.zeros(3, dtype=np.int32)
                    for mi in range(start_idx, target_idx + 1):
                        tv = tmean[y, mi, rr, cc]
                        pv = ppt[y, mi, rr, cc]
                        rv = srad[y, mi, rr, cc]
                        if math.isfinite(tv):
                            vals[0] += tv
                            cnts[0] += 1
                        if math.isfinite(pv):
                            vals[1] += pv
                            cnts[1] += 1
                        if math.isfinite(rv):
                            vals[2] += rv
                            cnts[2] += 1
                    if cnts[0] == 0 or cnts[1] == 0 or cnts[2] == 0:
                        continue
                    data[n, 0] = yv
                    data[n, 1] = vals[0] / cnts[0]
                    data[n, 2] = vals[1] / cnts[1]
                    data[n, 3] = vals[2] / cnts[2]
                    n += 1
                if n < min_n:
                    continue
                score = partial_corr_score(data, n, 4)
                if score > best_score:
                    best_score = score
                    best_lag = lag
            if best_lag > 0:
                lag_out[rr, cc] = best_lag
                score_out[rr, cc] = best_score
    return (lag_out, score_out)

@nb.njit(cache=True, parallel=True)
def compute_opl_eos(pheno: np.ndarray, tmean: np.ndarray, ppt: np.ndarray, srad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (ny, h, w) = pheno.shape
    lag_out = np.full((h, w), OUT_NODATA, dtype=np.float32)
    score_out = np.full((h, w), OUT_NODATA, dtype=np.float32)
    min_n = int(math.ceil(ny * THRESHOLD))
    for rr in nb.prange(h):
        data = np.empty((ny, 4), dtype=np.float64)
        for cc in range(w):
            nph = 0
            ymean = 0.0
            for y in range(ny):
                val = pheno[y, rr, cc]
                if math.isfinite(val):
                    nph += 1
                    ymean += val
            if nph < min_n:
                continue
            ymean /= nph
            month = doy_to_month_numba(ymean)
            if month < 1:
                continue
            best_lag = -1
            best_score = -1.0
            for lag in range(1, MAX_LAG + 1):
                n = 0
                for y in range(ny):
                    yv = pheno[y, rr, cc]
                    if not math.isfinite(yv):
                        continue
                    end_idx = (y + 1) * 12 + (month - 1)
                    start_idx = end_idx - lag + 1
                    if start_idx < 0:
                        start_idx = 0
                    vals = np.zeros(3, dtype=np.float64)
                    cnts = np.zeros(3, dtype=np.int32)
                    for mi in range(start_idx, end_idx + 1):
                        tv = tmean[mi, rr, cc]
                        pv = ppt[mi, rr, cc]
                        rv = srad[mi, rr, cc]
                        if math.isfinite(tv):
                            vals[0] += tv
                            cnts[0] += 1
                        if math.isfinite(pv):
                            vals[1] += pv
                            cnts[1] += 1
                        if math.isfinite(rv):
                            vals[2] += rv
                            cnts[2] += 1
                    if cnts[0] == 0 or cnts[1] == 0 or cnts[2] == 0:
                        continue
                    data[n, 0] = yv
                    data[n, 1] = vals[0] / cnts[0]
                    data[n, 2] = vals[1] / cnts[1]
                    data[n, 3] = vals[2] / cnts[2]
                    n += 1
                if n < min_n:
                    continue
                score = partial_corr_score(data, n, 4)
                if score > best_score:
                    best_score = score
                    best_lag = lag
            if best_lag > 0:
                lag_out[rr, cc] = best_lag
                score_out[rr, cc] = best_score
    return (lag_out, score_out)

@nb.njit(cache=True, parallel=True)
def mean_doy(pheno: np.ndarray) -> np.ndarray:
    (ny, h, w) = pheno.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    for rr in nb.prange(h):
        for cc in range(w):
            s = 0.0
            n = 0
            for y in range(ny):
                v = pheno[y, rr, cc]
                if math.isfinite(v):
                    s += v
                    n += 1
            if n > 0:
                out[rr, cc] = s / n
    return out

@nb.njit(cache=True, parallel=True)
def extract_preseason_sos(opl: np.ndarray, mean_doy_arr: np.ndarray, climate: np.ndarray, mode_sum: bool) -> np.ndarray:
    (ny, _, h, w) = climate.shape
    out = np.full((ny, h, w), OUT_NODATA, dtype=np.float32)
    for rr in nb.prange(h):
        for cc in range(w):
            lag = int(round(opl[rr, cc]))
            if lag <= 0 or lag > 12:
                continue
            month = doy_to_month_numba(mean_doy_arr[rr, cc])
            if month < 1:
                continue
            if month > 6:
                month = 6
            end_idx = month + 2
            start_idx = end_idx - lag + 1
            if start_idx < 0:
                start_idx = 0
            for y in range(ny):
                s = 0.0
                n = 0
                for mi in range(start_idx, end_idx + 1):
                    v = climate[y, mi, rr, cc]
                    if math.isfinite(v):
                        s += v
                        n += 1
                if n > 0:
                    out[y, rr, cc] = s if mode_sum else s / n
    return out

@nb.njit(cache=True, parallel=True)
def extract_preseason_eos(opl: np.ndarray, mean_doy_arr: np.ndarray, climate: np.ndarray, mode_sum: bool) -> np.ndarray:
    (nmonths, h, w) = climate.shape
    ny = 20
    out = np.full((ny, h, w), OUT_NODATA, dtype=np.float32)
    for rr in nb.prange(h):
        for cc in range(w):
            lag = int(round(opl[rr, cc]))
            if lag <= 0 or lag > 12:
                continue
            month = doy_to_month_numba(mean_doy_arr[rr, cc])
            if month < 1:
                continue
            for y in range(ny):
                end_idx = (y + 1) * 12 + (month - 1)
                if end_idx < 0 or end_idx >= nmonths:
                    continue
                start_idx = end_idx - lag + 1
                if start_idx < 0:
                    start_idx = 0
                s = 0.0
                n = 0
                for mi in range(start_idx, end_idx + 1):
                    v = climate[mi, rr, cc]
                    if math.isfinite(v):
                        s += v
                        n += 1
                if n > 0:
                    out[y, rr, cc] = s if mode_sum else s / n
    return out

def run_opl(dirs: dict[str, Path], overwrite: bool, log_path: Path) -> None:
    ref = reference_path()
    profile = output_profile_from_ref(ref, 'float32', 1, OUT_NODATA)
    with rasterio.open(ref) as src:
        (width, height) = (src.width, src.height)
    for pheno in PHENO_TARGETS:
        lag_p = opl_lag_path(pheno, dirs)
        score_p = opl_score_path(pheno, dirs)
        if lag_p.exists() and score_p.exists() and (not overwrite):
            log_line(log_path, f'Skip existing OPL {pheno}')
            continue
        log_line(log_path, f'Start OPL {pheno}')
        with rasterio.open(lag_p, 'w', **profile) as lag_dst, rasterio.open(score_p, 'w', **profile) as score_dst:
            lag_dst.set_band_description(1, f'{pheno}_OPL_months')
            score_dst.set_band_description(1, f'{pheno}_OPL_abs_partial_corr_tmean')
            windows = list(iter_windows(width, height))
            for win in tqdm(windows, desc=f'OPL {pheno}'):
                pheno_stack = read_pheno_stack(pheno, win, dirs)
                if not np.isfinite(pheno_stack).any():
                    empty = np.full((int(win.height), int(win.width)), OUT_NODATA, dtype=np.float32)
                    lag_dst.write(empty, 1, window=win)
                    score_dst.write(empty, 1, window=win)
                    continue
                if pheno == 'sos':
                    clim = [read_sos_climate_stack(v, win) for v in OPL_VARS]
                    (lag, score) = compute_opl_sos(pheno_stack, clim[0], clim[1], clim[2])
                else:
                    clim = [read_eos_climate_stack(v, win) for v in OPL_VARS]
                    (lag, score) = compute_opl_eos(pheno_stack, clim[0], clim[1], clim[2])
                lag_dst.write(lag, 1, window=win)
                score_dst.write(score, 1, window=win)
                del pheno_stack, clim, lag, score
                gc.collect()
        log_line(log_path, f'Done OPL {pheno}')

def scan_climate_vars() -> list[str]:
    vars_found = set()
    for p in CLIMATE_DIR.glob('TerraClimate_*_????_M??_aligned_mask.tif'):
        m = re.match('TerraClimate_(.+?)_\\d{4}_M\\d{2}_aligned_mask\\.tif$', p.name)
        if m:
            vars_found.add(m.group(1))
    return sorted(vars_found)

def run_preseason(dirs: dict[str, Path], overwrite: bool, log_path: Path, selected_vars: list[str] | None=None, selected_phenos: list[str] | None=None) -> None:
    ref = reference_path()
    profile = output_profile_from_ref(ref, 'float32', len(PHENO_YEARS), OUT_NODATA)
    with rasterio.open(ref) as src:
        (width, height) = (src.width, src.height)
    climate_vars = scan_climate_vars()
    if selected_vars:
        selected = set(selected_vars)
        climate_vars = [v for v in climate_vars if v in selected]
    phenos = selected_phenos if selected_phenos else PHENO_TARGETS
    log_line(log_path, 'Climate vars: ' + ', '.join(climate_vars))
    for pheno in phenos:
        pheno_profile_path = pheno_doy_path(pheno, PHENO_YEARS[0], dirs)
        for var in climate_vars:
            out_p = preseason_path(pheno, var, dirs)
            if out_p.exists() and (not overwrite):
                continue
            mode_sum = var in SUM_VARS
            log_line(log_path, f"Start preseason {pheno} {var} {('sum' if mode_sum else 'mean')}")
            with rasterio.open(out_p, 'w', **profile) as dst:
                for (i, year) in enumerate(PHENO_YEARS, start=1):
                    dst.set_band_description(i, f"Preseason_{pheno}_{var}_{('sum' if mode_sum else 'mean')}_{year}")
                with rasterio.open(opl_lag_path(pheno, dirs)) as opl_src:
                    windows = list(iter_windows(width, height))
                    for win in tqdm(windows, desc=f'Preseason {pheno} {var}'):
                        opl = opl_src.read(1, window=win).astype(np.float32, copy=False)
                        opl[opl == opl_src.nodata] = np.nan
                        if not np.isfinite(opl).any():
                            empty = np.full((int(win.height), int(win.width)), OUT_NODATA, dtype=np.float32)
                            for band in range(1, len(PHENO_YEARS) + 1):
                                dst.write(empty, band, window=win)
                            continue
                        pheno_stack = read_pheno_stack(pheno, win, dirs)
                        mdoy = mean_doy(pheno_stack)
                        if pheno == 'sos':
                            climate = read_sos_climate_stack(var, win)
                            result = extract_preseason_sos(opl, mdoy, climate, mode_sum)
                        else:
                            climate = read_eos_climate_stack(var, win)
                            result = extract_preseason_eos(opl, mdoy, climate, mode_sum)
                        result[~np.isfinite(result)] = OUT_NODATA
                        for band in range(result.shape[0]):
                            dst.write(result[band], band + 1, window=win)
                        del pheno_stack, mdoy, climate, result
                        gc.collect()
                dst.update_tags(reference=str(pheno_profile_path), statistic='sum' if mode_sum else 'mean')
            log_line(log_path, f'Done preseason {pheno} {var}: {out_p}')

def read_preseason_stack(pheno: str, var: str, win: Window, dirs: dict[str, Path]) -> np.ndarray:
    p = preseason_path(pheno, var, dirs)
    with rasterio.open(p) as src:
        arr = src.read(window=win).astype(np.float32, copy=False)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr

def calculate_sensitivity_tile(y_arr: np.ndarray, t_arr: np.ndarray, p_arr: np.ndarray, r_arr: np.ndarray) -> np.ndarray:
    (ny, h, w) = y_arr.shape
    n_pix = h * w
    y = y_arr.reshape(ny, n_pix).T
    t = t_arr.reshape(ny, n_pix).T
    p = p_arr.reshape(ny, n_pix).T
    r = r_arr.reshape(ny, n_pix).T
    valid = np.isfinite(y) & np.isfinite(t) & np.isfinite(p) & np.isfinite(r)
    n_valid = valid.sum(axis=1)
    use = n_valid >= MIN_VALID_YEARS
    out = np.full((9, n_pix), OUT_NODATA, dtype=np.float32)
    if not np.any(use):
        return out.reshape(9, h, w)
    idx_all = np.flatnonzero(use)
    yk = np.where(valid[use], y[use], 0.0).astype(np.float32)
    tk = np.where(valid[use], t[use], 0.0).astype(np.float32)
    pk = np.where(valid[use], p[use], 0.0).astype(np.float32)
    rk = np.where(valid[use], r[use], 0.0).astype(np.float32)
    mask = valid[use]
    n = n_valid[use].astype(np.float32)

    def center(a: np.ndarray) -> np.ndarray:
        mu = a.sum(axis=1) / n
        return np.where(mask, a - mu[:, None], 0.0)

    def minmax(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                                                                                      
        amin = np.min(np.where(mask, a, np.inf), axis=1)
        amax = np.max(np.where(mask, a, -np.inf), axis=1)
        span = amax - amin
        safe_span = np.where(span > 0, span, 1.0)
        normalized = np.where(mask, (a - amin[:, None]) / safe_span[:, None], 0.0)
        return (normalized.astype(np.float32), span.astype(np.float32))
    (yc, tc, pc, rc) = (center(yk), center(tk), center(pk), center(rk))
    (yn, yspan) = minmax(yk)
    (tn, tspan) = minmax(tk)
    (pn, pspan) = minmax(pk)
    (rn, rspan) = minmax(rk)
    nonzero = (np.sum(yc * yc, axis=1) > 0) & (np.sum(tc * tc, axis=1) > 0) & (np.sum(pc * pc, axis=1) > 0) & (np.sum(rc * rc, axis=1) > 0) & (yspan > 0) & (tspan > 0) & (pspan > 0) & (rspan > 0)
    if not np.any(nonzero):
        return out.reshape(9, h, w)
    idx = idx_all[nonzero]
    (yc, tc, pc, rc) = (yc[nonzero], tc[nonzero], pc[nonzero], rc[nonzero])
    (yn, tn, pn, rn) = (yn[nonzero], tn[nonzero], pn[nonzero], rn[nonzero])
    mask2 = mask[nonzero]
    n2 = n[nonzero]
    yn_mean = yn.sum(axis=1) / n2
    xn = np.stack([tn, pn, rn], axis=-1)
    xn_mean = xn.sum(axis=1) / n2[:, None]
    yn_centered = np.where(mask2, yn - yn_mean[:, None], 0.0)
    xn_centered = np.where(mask2[:, :, None], xn - xn_mean[:, None, :], 0.0)
    xn_std = np.sqrt(np.sum(xn_centered * xn_centered, axis=1) / np.maximum(n2[:, None] - 1.0, 1.0))
    x_standardized = xn_centered / xn_std[:, None, :]
    xtx = np.einsum('pni,pnj->pij', x_standardized, x_standardized)
    xty = np.einsum('pni,pn->pi', x_standardized, yn_centered)
    ridge = xtx + RIDGE_ALPHA * np.eye(3, dtype=np.float32)
    try:
        standardized_coefs = np.linalg.solve(ridge, xty[:, :, None]).squeeze(-1)
    except np.linalg.LinAlgError:
        standardized_coefs = np.einsum('pij,pj->pi', np.linalg.pinv(ridge), xty)
    coefs = standardized_coefs / xn_std
    z = np.stack([yc, tc, pc, rc], axis=-1)
    s = np.einsum('pni,pnj->pij', z, z)
    s += 1e-08 * np.eye(4, dtype=np.float32)
    try:
        invs = np.linalg.inv(s)
    except np.linalg.LinAlgError:
        invs = np.linalg.pinv(s)
    denom_t = invs[:, 0, 0] * invs[:, 1, 1]
    denom_p = invs[:, 0, 0] * invs[:, 2, 2]
    denom_r = invs[:, 0, 0] * invs[:, 3, 3]
    pr_t = np.full(len(idx), np.nan, dtype=np.float32)
    pr_p = np.full(len(idx), np.nan, dtype=np.float32)
    pr_r = np.full(len(idx), np.nan, dtype=np.float32)
    good = denom_t > 0
    pr_t[good] = -invs[good, 0, 1] / np.sqrt(denom_t[good])
    good = denom_p > 0
    pr_p[good] = -invs[good, 0, 2] / np.sqrt(denom_p[good])
    good = denom_r > 0
    pr_r[good] = -invs[good, 0, 3] / np.sqrt(denom_r[good])
    pr_t = np.clip(pr_t, -0.99999, 0.99999)
    pr_p = np.clip(pr_p, -0.99999, 0.99999)
    pr_r = np.clip(pr_r, -0.99999, 0.99999)
    df = n2 - 4
    pval_t = np.full_like(pr_t, np.nan)
    pval_p = np.full_like(pr_p, np.nan)
    pval_r = np.full_like(pr_r, np.nan)
    good_df = df > 0
    pval_t[good_df] = student_t.sf(np.abs(pr_t[good_df] * np.sqrt(df[good_df] / (1 - pr_t[good_df] ** 2))), df[good_df]) * 2
    pval_p[good_df] = student_t.sf(np.abs(pr_p[good_df] * np.sqrt(df[good_df] / (1 - pr_p[good_df] ** 2))), df[good_df]) * 2
    pval_r[good_df] = student_t.sf(np.abs(pr_r[good_df] * np.sqrt(df[good_df] / (1 - pr_r[good_df] ** 2))), df[good_df]) * 2
    vals = np.column_stack([coefs[:, 0], coefs[:, 1], coefs[:, 2], pr_t, pr_p, pr_r, pval_t, pval_p, pval_r]).astype(np.float32)
    vals[~np.isfinite(vals)] = OUT_NODATA
    for i in range(9):
        out[i, idx] = vals[:, i]
    return out.reshape(9, h, w)

def run_sensitivity(dirs: dict[str, Path], overwrite: bool, log_path: Path, selected_phenos: list[str] | None=None) -> None:
    ref = reference_path()
    profile = output_profile_from_ref(ref, 'float32', 9, OUT_NODATA)
    with rasterio.open(ref) as src:
        (width, height) = (src.width, src.height)
    phenos = selected_phenos if selected_phenos else PHENO_TARGETS
    for pheno in phenos:
        out_p = sensitivity_path(pheno, dirs)
        if out_p.exists() and (not overwrite):
            continue
        log_line(log_path, f'Start sensitivity {pheno}')
        with rasterio.open(out_p, 'w', **profile) as dst:
            for (i, desc) in enumerate(BAND_DESCRIPTIONS, start=1):
                dst.set_band_description(i, desc)
            dst.update_tags(ridge_method=RIDGE_METHOD, ridge_parameter=str(RIDGE_ALPHA), ridge_input_scaling='pixel-wise min-max normalization of phenology and all predictors', ridge_coefficient_units='dimensionless', method_reference='Wang et al. 2022 Nature Climate Change; doi:10.1038/s41558-022-01285-w')
            windows = list(iter_windows(width, height))
            for win in tqdm(windows, desc=f'Sensitivity {pheno}'):
                y = read_pheno_stack(pheno, win, dirs)
                if not np.isfinite(y).any():
                    dst.write(np.full((9, int(win.height), int(win.width)), OUT_NODATA, dtype=np.float32), window=win)
                    continue
                t = read_preseason_stack(pheno, 'tmean', win, dirs)
                p = read_preseason_stack(pheno, 'ppt', win, dirs)
                r = read_preseason_stack(pheno, 'srad', win, dirs)
                result = calculate_sensitivity_tile(y, t, p, r)
                dst.write(result, window=win)
                del y, t, p, r, result
                gc.collect()
        log_line(log_path, f'Done sensitivity {pheno}: {out_p}')

def write_inventory(dirs: dict[str, Path]) -> None:
    rows = []
    for pheno in PHENO_TARGETS:
        for year in PHENO_YEARS:
            rows.append({'type': 'phenology_source', 'name': f'{pheno}_{year}', 'path': str(pheno_source_path(pheno, year)), 'exists': pheno_source_path(pheno, year).exists()})
    for var in scan_climate_vars():
        for year in CLIMATE_YEARS:
            for month in range(1, 13):
                p = climate_path(var, year, month)
                rows.append({'type': 'climate', 'name': f'{var}_{year}_M{month:02d}', 'path': str(p), 'exists': p.exists()})
    out = dirs['logs'] / 'pipeline_input_inventory.csv'
    with out.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'name', 'path', 'exists'])
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run MODIS phenology preseason and sensitivity pipeline on the wind grid.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs.')
    parser.add_argument('--steps', nargs='+', choices=['phenology', 'opl', 'preseason', 'sensitivity', 'all'], default=['all'], help='Pipeline steps to run.')
    parser.add_argument('--vars', nargs='+', default=None, help='Climate variables to use for the preseason step. Defaults to all scanned variables.')
    parser.add_argument('--phenos', nargs='+', choices=PHENO_TARGETS, default=None, help='Phenology targets to use for the preseason step. Defaults to sos eos.')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    dirs = ensure_dirs()
    log_path = dirs['logs'] / 'run_preseason_pipeline_wind_grid.log'
    steps = set(args.steps)
    if 'all' in steps:
        steps = {'phenology', 'opl', 'preseason', 'sensitivity'}
    write_inventory(dirs)
    log_line(log_path, f"Steps: {' '.join(sorted(steps))}")
    log_line(log_path, f'Years: {PHENO_YEARS[0]}-{PHENO_YEARS[-1]}')
    log_line(log_path, f'Output root: {OUT_ROOT}')
    if 'phenology' in steps:
        create_pheno_doy(dirs, args.overwrite, log_path)
    if 'opl' in steps:
        run_opl(dirs, args.overwrite, log_path)
    if 'preseason' in steps:
        run_preseason(dirs, args.overwrite, log_path, args.vars, args.phenos)
    if 'sensitivity' in steps:
        run_sensitivity(dirs, args.overwrite, log_path, args.phenos)
    log_line(log_path, 'Pipeline finished')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
