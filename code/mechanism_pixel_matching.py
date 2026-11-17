                                                                                    
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.workspace.is_dir():
        parser.error('The prepared original workspace must exist')
    specification = json.loads(arguments.job.read_text(encoding='utf8'))
    allowed_jobs = {'psm_prepare', 'match', 'psm_summary'}
    if specification.get('kind') not in allowed_jobs:
        parser.error('Use an original psm_prepare, match, or psm_summary job')
    os.environ['MORPH_PIXEL_PSM_WORKSPACE'] = str(arguments.workspace.resolve())
    sys.argv = ['pixel_psm_original_worker.py', str(arguments.job.resolve())]
    worker = Path(__file__).with_name('pixel_psm_original_worker.py')
    runpy.run_path(str(worker), run_name='__main__')
if __name__ == '__main__':
    main()
