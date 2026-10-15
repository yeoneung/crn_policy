"""Fetch and checksum-verify the original data for a Git checkout."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    legacy = ROOT / 'reproducibility' / 'crn_policy'
    subprocess.run([sys.executable, str(legacy / 'download_data.py')],
                   cwd=legacy, check=True)
