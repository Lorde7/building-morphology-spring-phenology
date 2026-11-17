                                                                 
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='The full SEM entry is code/mechanism_sem.py and requires R with lavaan.',
    )
    parser.add_argument('--output', type=Path, required=True, help='Output figure path')
    args = parser.parse_args()
    d = pd.read_csv(ROOT / 'data' / 'station_pairs.csv')
    assert len(d) == 373 and np.allclose(d.high_wind-d.low_wind, d.delta_wind)
    fit = sm.OLS(d.delta_wind, np.ones((len(d), 1))).fit(
        cov_type='cluster', cov_kwds={'groups': d.pair_pixel_cluster})
    ci = np.asarray(fit.conf_int())[0]
    print(f'Pairs: {len(d)}; high - low: {fit.params.iloc[0]:.6f} m/s; '
          f'95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]; P: {fit.pvalues.iloc[0]:.6g}')
    fig, ax = plt.subplots(figsize=(3.5, 4))
    for low, high in zip(d.low_wind, d.high_wind):
        ax.plot([0, 1], [low, high], color='0.7', alpha=.15, linewidth=.4)
    ax.boxplot([d.low_wind, d.high_wind], positions=[0, 1], showfliers=False)
    ax.set_xticks([0, 1], ['Low BMSI', 'High BMSI'])
    ax.set_ylabel('Observed preseason wind speed (m/s)')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()
