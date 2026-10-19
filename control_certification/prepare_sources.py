"""Create an isolated workspace for the preserved simulation sources.

fresh: source inputs only, for new training/simulation.
records: sources plus all original observations, for inspection or replay.
coupling: sources, fixed checkpoints and prerequisite stable/calibration data,
          leaving new coupling-study output cells absent for regeneration.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse
import importlib.util
import shutil

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'reproducibility/crn_policy'


def prepare(output, mode):
    spec = importlib.util.spec_from_file_location('frozen_workspace', LEGACY / 'reproduce.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = runner.read_manifest()
    runner.verify_frozen(manifest)
    data = LEGACY / 'data' / manifest['data_archive']['filename']
    if mode != 'fresh':
        runner.verify_archive(data, manifest)
    runner.stage(output, manifest, data if mode == 'records' else None)
    sources = ROOT / 'supplementary_records/original_scripts'
    shutil.copyfile(sources / 'mechanical_physical_metrics.py',
                    output / 'code/scripts/mechanical_physical_metrics.py')
    shutil.copyfile(sources / 'MECHANICAL_PHYSICAL_METRICS_PLAN.md',
                    output / 'MECHANICAL_PHYSICAL_METRICS_PLAN.md')
    if mode == 'coupling':
        prefix = 'code/results/v2_coupling_sensitivity/'
        with ZipFile(data) as archive:
            for name in manifest['data_files']:
                prerequisite = name.startswith(('code/results/coupled_support_confirmatory/',
                                                 'code/results/stability_calibration_confirmatory/'))
                checkpoint = name.startswith(prefix + 'frozen_policies/')
                protocol = name in {prefix + n for n in ['protocol.json', 'source_hashes.json', 'input_hashes.json']}
                if prerequisite or checkpoint or protocol:
                    target = runner.contained_path(output, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
    print(f'Prepared {mode} source workspace: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory, never overwritten.')
    parser.add_argument('--mode', choices=['fresh', 'records', 'coupling'], default='fresh')
    args = parser.parse_args()
    prepare(args.output.resolve(), args.mode)
