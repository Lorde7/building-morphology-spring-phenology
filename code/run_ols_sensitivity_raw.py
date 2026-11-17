                                                                 

                                                                            
                                                                            
                                                                              
                                                                             
      
   
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm
import run_preseason_pipeline_wind_grid as base
OLS_BANDS = ['OLS_T', 'OLS_P', 'OLS_R', 'PartialCorr_T', 'PartialCorr_P', 'PartialCorr_R', 'Pval_T', 'Pval_P', 'Pval_R']

def ols_coefficients_tile(y_arr: np.ndarray, t_arr: np.ndarray, p_arr: np.ndarray, r_arr: np.ndarray, min_valid_years: int) -> np.ndarray:
                                                                              
    (ny, h, w) = y_arr.shape
    n_pix = h * w
    arrays = [a.reshape(ny, n_pix).T for a in (y_arr, t_arr, p_arr, r_arr)]
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    n_valid = valid.sum(axis=1)
    use = n_valid >= min_valid_years
    out = np.full((3, n_pix), base.OUT_NODATA, dtype=np.float32)
    if not np.any(use):
        return out.reshape(3, h, w)
    selected = [np.where(valid[use], a[use], 0.0).astype(np.float64) for a in arrays]
    mask = valid[use]
    n = n_valid[use].astype(np.float64)
    centered = []
    for a in selected:
        mean = a.sum(axis=1) / n
        centered.append(np.where(mask, a - mean[:, None], 0.0))
    (yc, tc, pc, rc) = centered
    nonzero = np.logical_and.reduce([np.sum(a * a, axis=1) > 0 for a in (yc, tc, pc, rc)])
    if not np.any(nonzero):
        return out.reshape(3, h, w)
    x = np.stack([tc[nonzero], pc[nonzero], rc[nonzero]], axis=-1)
    yy = yc[nonzero]
    xtx = np.einsum('pni,pnj->pij', x, x)
    xty = np.einsum('pni,pn->pi', x, yy)
    beta = np.einsum('pij,pj->pi', np.linalg.pinv(xtx, rcond=1e-10), xty)
    idx = np.flatnonzero(use)[nonzero]
    good = np.all(np.isfinite(beta), axis=1)
    out[:, idx[good]] = beta[good].T.astype(np.float32)
    return out.reshape(3, h, w)

def calculate_ols_tile(y, t, p, r, min_valid_years):
    result = base.calculate_sensitivity_tile(y, t, p, r)
    result[:3] = ols_coefficients_tile(y, t, p, r, min_valid_years)
    invalid = result[0] == base.OUT_NODATA
    result[:, invalid] = base.OUT_NODATA
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pheno', choices=['sos', 'eos'], default='sos')
    parser.add_argument('--min-valid-years', type=int, default=11)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    dirs = base.ensure_dirs()
    out = dirs['sensitivity'] / f'olsAnalysis_{args.pheno}_9bands_raw_units_min{args.min_valid_years}.tif'
    if out.exists() and (not args.overwrite):
        raise FileExistsError(f'Refusing to overwrite existing OLS output: {out}')
    ref = base.reference_path()
    profile = base.output_profile_from_ref(ref, 'float32', 9, base.OUT_NODATA)
    with rasterio.open(ref) as src:
        (width, height) = (src.width, src.height)
    with rasterio.open(out, 'w', **profile) as dst:
        for (i, desc) in enumerate(OLS_BANDS, start=1):
            dst.set_band_description(i, desc)
        dst.update_tags(method='complete-case multivariable OLS with intercept', response='annual phenological date in day of year', predictors='preseason mean temperature, total precipitation, total solar radiation', input_scaling='none; regression fitted in original units after implicit mean-centering', temperature_coefficient_units='days per degree C', temperature_sign='negative means earlier phenology under warmer preseason conditions', minimum_valid_years=str(args.min_valid_years), comparison_reference='Shen et al. 2024 Nature Climate Change; doi:10.1038/s41558-024-02035-w')
        windows = list(base.iter_windows(width, height))
        for win in tqdm(windows, desc=f'Raw-unit OLS {args.pheno}'):
            y = base.read_pheno_stack(args.pheno, win, dirs)
            if not np.isfinite(y).any():
                empty = np.full((9, int(win.height), int(win.width)), base.OUT_NODATA, dtype=np.float32)
                dst.write(empty, window=win)
                continue
            t = base.read_preseason_stack(args.pheno, 'tmean', win, dirs)
            p = base.read_preseason_stack(args.pheno, 'ppt', win, dirs)
            r = base.read_preseason_stack(args.pheno, 'srad', win, dirs)
            result = calculate_ols_tile(y, t, p, r, args.min_valid_years)
            dst.write(result, window=win)
            del y, t, p, r, result
            gc.collect()
    print(out)
if __name__ == '__main__':
    main()
