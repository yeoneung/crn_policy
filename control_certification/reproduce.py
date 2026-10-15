"""Control certification runner (Python 3.11+, CPU, no network).

verify: check packaged file hashes and all frozen source/data records.
quick: tests, finite-power stress checks and fresh nonnormal example.
all: quick plus recorded summary/figure replay and every legacy replay group.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'reproducibility/crn_policy'


def verify_package():
    manifest = ROOT / 'CODE_MANIFEST.json'
    if not manifest.exists():
        print('Working tree: release manifest not present; frozen hashes still checked.', flush=True)
        return
    records = json.loads(manifest.read_text(encoding='utf-8'))['files']
    for relative, record in records.items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Invalid release manifest path')
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing or wrong-sized release file: {relative}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'Release file hash mismatch: {relative}')
    print(f'Verified {len(records)} package files.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'quick', 'all'])
    parser.add_argument('--work-dir', type=Path, help='New directory for logs and generated evidence')
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work = (args.work_dir or ROOT / 'runs' / f'{args.command}-{timestamp}').resolve()
    work.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ['numpy', 'scipy', 'matplotlib', 'threadpoolctl', 'gymnasium']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    record = {'command': args.command, 'python': sys.version, 'platform': sys.platform,
              'dependencies': versions, 'commands': [], 'status': 'running',
              'scope': 'Fresh finite-power demonstration plus replay, not full benchmark retraining.'}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def execute(label, arguments, cwd=ROOT):
        print(f'[{label}]', flush=True)
        log = work / f'{label}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd,
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        record['commands'].append({'label': label, 'arguments': list(map(str, arguments)),
                                   'exit_code': result.returncode, 'log': log.name})
        if result.returncode:
            print(log.read_text(encoding='utf-8', errors='replace')[-6000:])
            raise RuntimeError(f'{label} failed; see {log}')
        print(log.read_text(encoding='utf-8', errors='replace')[-1200:].rstrip(), flush=True)

    try:
        verify_package()
        execute('frozen_integrity', ['reproduce.py', 'verify'], LEGACY)
        if args.command in ['quick', 'all']:
            execute('finite_power_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
            execute('legacy_runner_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'], LEGACY)
            execute('finite_power_stress', ['audit_matrix_certificate.py', '--output',
                                            work / 'evidence/matrix_certificate_audit.json'])
            execute('finite_power_example', ['-m', 'examples.nonnormal_certificate',
                                             '--output', work / 'finite_power_example.json'])
        if args.command == 'quick':
            execute('legacy_verifier_tests', ['reproduce.py', 'tests', '--work-dir', work / 'legacy-tests'], LEGACY)
        if args.command == 'all':
            execute('recorded_summaries', ['audit_evidence.py', '--output',
                                          work / 'evidence/numerical_audit.json'])
            execute('finite_power_study', ['-m', 'examples.finite_power_study',
                                           '--replay-from', ROOT / 'evidence/finite_power_study',
                                           '--output-dir', work / 'finite-power-study'])
            execute('finite_power_study_audit', ['audit_finite_power_study.py', '--output',
                                                 work / 'evidence/finite_power_study_audit.json'])
            execute('summary_figure', ['make_figure.py', '--output-dir', work / 'figure'])
            execute('legacy_all', ['reproduce.py', 'all', '--work-dir', work / 'legacy-all'], LEGACY)
        record['status'] = 'passed'
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        (work / 'run_report.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f'Passed: {work}', flush=True)


if __name__ == '__main__':
    main()
