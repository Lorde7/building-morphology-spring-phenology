                                                             
import json
import os
from pathlib import Path

configuration_file = os.environ.get('MORPH_INPUT_CONFIG')
if not configuration_file:
    raise RuntimeError('Set MORPH_INPUT_CONFIG to your input-path JSON file')
configuration = json.loads(Path(configuration_file).read_text(encoding='utf8'))
ROOT = Path(configuration['collect_CMIP6_ST_stores_20261012.path1'])
OUT = ROOT / 'CMIP6_same_period_20261012'
