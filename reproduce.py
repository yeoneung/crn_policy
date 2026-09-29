"""Portable entry point for frozen experiment code (Python 3.11+).

Each execution gets a new workspace. The recorded data and original scripts
are verified before use, and analysis never writes into the checked-in sources.
The runner itself uses only the Python standard library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
GROUPS = {
    'tests': [('-m', 'unittest', 'test_variance_replay', 'test_coupling_sensitivity', '-v')],
    'support': [('reproduce_main_evidence.py',)],
    'variance': [('variance_replay.py',)],
    'coupling': [('summarize_coupling_sensitivity.py',), ('audit_coupling_sensitivity.py',)],
    'pendulum': [('summarize_pendulum_regional.py',), ('audit_pendulum_regional.py',),
                 ('audit_nominal_reference.py',), ('audit_oracle_reference.py',)],
}
DATA_GROUPS = {'support', 'variance', 'coupling', 'pendulum', 'all'}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    """Accept only portable relative regular-file names inside root."""
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe archive path: {relative!r}')
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves workspace: {relative!r}')
    return target


def read_manifest():
    return json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))


def verify_frozen(manifest):
    for relative, expected in manifest['frozen_files'].items():
        path = contained_path(ROOT / 'frozen', relative)
        if not path.is_file() or digest(path) != expected['sha256']:
            raise ValueError(f'Frozen source missing or changed: {relative}')
    print(f"Verified {len(manifest['frozen_files'])} frozen inputs.", flush=True)


def verify_archive(path: Path, manifest):
    if not path.is_file():
        raise FileNotFoundError(f'Recorded data not found: {path}. See data/README.md.')
    if digest(path) != manifest['data_archive']['sha256']:
        raise ValueError('Recorded archive SHA-256 mismatch; use the matching release asset.')
    with ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['data_files']):
            raise ValueError('Recorded archive member list differs from the manifest.')
        for name in names:
            contained_path(ROOT / 'runs', name)
            blob = z.read(name)
            expected = manifest['data_files'][name]
            if len(blob) != expected['bytes'] or hashlib.sha256(blob).hexdigest() != expected['sha256']:
                raise ValueError(f'Recorded data changed: {name}')
    print(f"Verified {len(manifest['data_files'])} recorded files.", flush=True)


def stage(work: Path, manifest, data: Path | None):
    if work.exists():
        raise FileExistsError(f'Workspace already exists: {work}. Choose a new --work-dir.')
    work.mkdir(parents=True)
    for relative in manifest['frozen_files']:
        dest = contained_path(work, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'frozen' / relative, dest)
    if data:
        with ZipFile(data) as z:
            for name in manifest['data_files']:
                dest = contained_path(work, name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(name))
    for name in ['manuscript/tables', 'manuscript/figs_revision', 'code/results', 'logs']:
        (work / name).mkdir(parents=True, exist_ok=True)


def environment_record():
    versions = {}
    for name in ['numpy', 'scipy', 'matplotlib', 'threadpoolctl', 'gymnasium']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version.split()[0], 'platform': sys.platform, 'dependencies': versions}


def execute(group, work, record):
    env = os.environ.copy()
    env.update({'PYTHONUTF8': '1', 'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    for index, args in enumerate(GROUPS[group], 1):
        command = [sys.executable, *args]
        print(f"[{group}] {' '.join(args)}", flush=True)
        start = time.perf_counter()
        log = work / 'logs' / f'{group}_{index}.txt'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=work / 'code/scripts', env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        record['commands'].append({'group': group, 'args': list(args),
                                   'exit_code': result.returncode,
                                   'elapsed_seconds': round(time.perf_counter() - start, 3),
                                   'log': log.relative_to(work).as_posix()})
        if result.returncode:
            print(log.read_text(encoding='utf-8', errors='replace')[-4000:])
            raise RuntimeError(f'{group} failed; see {log}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'verify', 'prepare', *GROUPS, 'all'])
    parser.add_argument('--data', type=Path, help='Path to crn_policy_experiment_data.zip')
    parser.add_argument('--work-dir', type=Path, help='New output directory; existing directories are never overwritten')
    parser.add_argument('--without-data', action='store_true', help='Prepare a fresh simulation workspace without recorded results')
    args = parser.parse_args(argv)
    if args.command == 'list':
        for name, commands in GROUPS.items():
            print(f"{name:10} " + '; '.join(' '.join(c) for c in commands))
        return 0
    if args.without_data and args.command != 'prepare':
        parser.error('--without-data is only supported with prepare')
    if args.without_data and args.data:
        parser.error('--without-data and --data cannot be used together')
    manifest = read_manifest()
    verify_frozen(manifest)
    data = args.data.resolve() if args.data else ROOT / 'data' / manifest['data_archive']['filename']
    needs_data = args.command in DATA_GROUPS or (args.command == 'prepare' and not args.without_data)
    use_data = needs_data or (args.command == 'verify' and (args.data is not None or data.is_file()))
    if use_data:
        verify_archive(data, manifest)
    if args.command == 'verify':
        if not use_data:
            print('Source-only verification; pass --data to verify the recorded banks too.')
        return 0
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work = args.work_dir.resolve() if args.work_dir else ROOT / 'runs' / f'{args.command}-{timestamp}'
    stage(work, manifest, data if use_data else None)
    record = {'command': args.command, 'environment': environment_record(),
              'input_archive_sha256': manifest['data_archive']['sha256'] if use_data else None,
              'commands': [], 'status': 'prepared'}
    try:
        if args.command != 'prepare':
            for group in (list(GROUPS) if args.command == 'all' else [args.command]):
                execute(group, work, record)
            record['status'] = 'passed'
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        generated = work / 'manuscript'
        if generated.exists() and args.command != 'prepare':
            for source, target in [('figs_revision', 'figures'), ('tables', 'tables')]:
                if (generated / source).exists():
                    shutil.copytree(generated / source, work / 'artifacts' / target)
        (work / 'run_report.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f"{record['status'].capitalize()}: {work}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)
