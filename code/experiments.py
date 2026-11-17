                                                                                    
from pathlib import Path
import argparse
import ast
import json
import os
import runpy
import sys
HERE = Path(__file__).resolve().parent
ENTRIES = {'1': 'experiment_figure1_partial_spearman.py', '2': 'experiment_figure2_xgboost_support.py', '3-psm': 'experiment_figure3_station_matching.py', '3-gam': 'experiment_figure3_gam.py', '4': 'experiment_figure4_city_annual.py'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=ENTRIES)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run', action='store_true')
    arguments = parser.parse_args()
    selected = ENTRIES
    if arguments.experiment:
        selected = {arguments.experiment: ENTRIES[arguments.experiment]}
    for (identifier, filename) in selected.items():
        ast.parse((HERE / filename).read_text(encoding='utf8'))
        print(identifier, filename, 'syntax OK')
    if not arguments.run:
        print('No scientific calculation executed. See ANALYSIS_NOTES.md for inputs and scope.')
        return
    if not arguments.experiment or not arguments.config:
        parser.error('--run requires --experiment and --config')
    config = json.loads(arguments.config.read_text(encoding='utf8'))
    if not isinstance(config, dict):
        parser.error('Configuration must be a JSON object of path keys')
    os.environ['MORPH_INPUT_CONFIG'] = str(arguments.config.resolve())
    script = HERE / ENTRIES[arguments.experiment]
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name='__main__')
if __name__ == '__main__':
    main()
