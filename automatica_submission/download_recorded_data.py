"""Fetch and verify the original released data before using this Git checkout.

The submission ZIP already contains these data and does not need this step.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    legacy = ROOT / 'reproducibility' / 'crn_policy'
    subprocess.run([sys.executable, str(legacy / 'download_data.py')],
                   cwd=legacy, check=True)
