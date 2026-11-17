                                                                                        
from pathlib import Path
import argparse
import shutil
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True, help='Original SEM archive with source_file.txt, models and standardized_parameters.csv')
    parser.add_argument('--rscript', default='Rscript')
    arguments = parser.parse_args()
    executable = shutil.which(arguments.rscript)
    if executable is None:
        parser.error('Rscript and lavaan are required')
    if not arguments.workdir.is_dir():
        parser.error('SEM working directory does not exist')
    script = Path(__file__).with_name('experiment_figure3_sem_original.R')
    result = subprocess.run([executable, str(script), str(arguments.workdir.resolve()), str(arguments.archive.resolve())], check=False)
    raise SystemExit(result.returncode)
if __name__ == '__main__':
    main()
