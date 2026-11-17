import os as _os, json as _json
from pathlib import Path as _Path

def configured_path(key):
    if not _os.environ.get('MORPH_INPUT_CONFIG'):
        raise RuntimeError('Configure MORPH_INPUT_CONFIG for archived experiment input/output paths: ' + key)
    config = _json.loads(_Path(_os.environ['MORPH_INPUT_CONFIG']).read_text(encoding='utf8'))
    if key not in config:
        raise RuntimeError('Missing configured path: ' + key)
    return config[key]
                                                                                                                                                                                                                                                  
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
ROOT = Path(__file__).resolve().parent
BASE = Path(configured_path('experiment_figure3_gam.path1'))
SOURCE = BASE / '01_现象/source_city_input_638.csv'
OUT = Path(__file__).resolve().parents[1] / 'outputs' / 'experiment_figure3_gam'
FEATURES = ['Preseason_sos_ws', 'BMSI_A', 'Preseason_sos_tmean', 'Preseason_sos_ppt', 'Preseason_sos_srad', 'Preseason_sos_vpd', 'lon_gpkg', 'lat_gpkg']
LAM = 0.1778279410038923
SEED = 42
REPS = 300

def terms():
    t = s(0, n_splines=8)
    for i in range(1, 8):
        t += s(i, n_splines=8)
    return t

def estimate(sample, reference, grid):
    scaler = StandardScaler().fit(sample[FEATURES])
    x = scaler.transform(sample[FEATURES])
    model = LinearGAM(terms(), lam=LAM).fit(x, sample.target.to_numpy(float))
    ref = scaler.transform(reference[FEATURES])
    mean_other = np.mean(model.predict(ref) - model.partial_dependence(term=0, X=ref))
    gx = np.zeros((len(grid), len(FEATURES)))
    gx[:, 0] = (grid - scaler.mean_[0]) / scaler.scale_[0]
    curve = mean_other + model.partial_dependence(term=0, X=gx)
    return (curve, model, scaler)

def resample(rep, data, grid):
    rng = np.random.default_rng(SEED + 7919 * (rep + 1))
    b = data.iloc[rng.integers(0, len(data), len(data))].reset_index(drop=True)
    return estimate(b, data, grid)[0]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    assert data.city_id.is_unique and len(data) == 638
    assert np.array_equal(data.target, data.ST_OLS_days_per_C)
    assert np.isfinite(data[['target', *FEATURES]]).all().all()
    old = pd.read_csv(BASE / '03_SEM过程草图/city_wind_GAM_original/original_city_input_639.csv')
    paired = data.merge(old, on='city_id', suffixes=('_primary', '_archived'))
    assert np.allclose(paired.target_archived, paired.target_old_Ridge_days_per_C)
    wind = data[FEATURES[0]].to_numpy(float)
    grid = np.linspace(*np.quantile(wind, [0.02, 0.98]), 120)
    q = np.quantile(wind, [0.1, 0.5, 0.9])
    (curve, model, scaler) = estimate(data, data, grid)
    literal = []
    for value in q:
        ref = data[FEATURES].copy()
        ref[FEATURES[0]] = value
        literal.append(float(np.mean(model.predict(scaler.transform(ref)))))
    fast = estimate(data, data, q)[0]
    assert np.max(np.abs(fast - literal)) < 1e-10
    boot = np.array(Parallel(n_jobs=4)((delayed(resample)(i, data, grid) for i in range(REPS))))
    (lo, hi) = np.quantile(boot, [0.025, 0.975], axis=0)
    points = np.interp(q, grid, curve)
    bp = np.array([np.interp(q, grid, c) for c in boot])
    (pl, ph) = np.quantile(bp, [0.025, 0.975], axis=0)
    pd.DataFrame({'wind_speed': grid, 'adjusted_ST': curve, 'ci_low': lo, 'ci_high': hi}).to_csv(OUT / 'curve.csv', index=False)
    pd.DataFrame({'level': ['Q10', 'Q50', 'Q90'], 'wind_speed': q, 'adjusted_ST': points, 'ci_low': pl, 'ci_high': ph}).to_csv(OUT / 'points.csv', index=False)
    pd.DataFrame({'term': FEATURES, 'P_approx': model.statistics_['p_values'][:8]}).to_csv(OUT / 'term_tests.csv', index=False)
    data[['city_id', 'target', *FEATURES]].to_csv(OUT / 'primary_input_638.csv', index=False)
    np.savez_compressed(OUT / 'bootstrap.npz', grid=grid, curves=boot, quantiles=q)
if __name__ == '__main__':
    main()
