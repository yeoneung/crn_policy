"""Build a self-contained code/data ZIP and SHA-256 manifest."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = 'control_certification.zip'
TOP = 'control_certification'


def payload_files():
    names = {'CODE_README.md', 'requirements.txt', 'LICENSE_NOTICE.md',
             'reproduce.py', 'package_code.py', 'download_recorded_data.py',
             'audit_evidence.py', 'audit_finite_power_study.py',
             'audit_matrix_certificate.py', 'make_figure.py'}
    for folder in ['certification', 'examples', 'tests', 'reproducibility',
                   'evidence/finite_power_study', 'evidence/finite_power_development']:
        for path in (ROOT / folder).rglob('*'):
            parts = path.relative_to(ROOT).parts
            if any(p in {'__pycache__', 'runs', '.git', '.venv', '.github'} for p in parts):
                continue
            if path.is_file() and path.name != '.DS_Store' and path.suffix != '.pyc':
                names.add(path.relative_to(ROOT).as_posix())
    return [ROOT / name for name in sorted(names)]


def main():
    files = payload_files()
    manifest = {'description': 'Control certification implementation and recorded simulation data.',
                'archive_root': TOP, 'contains_recorded_data': True,
                'full_benchmark_retraining_claimed': False, 'files': {}}
    for path in files:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['files'][path.relative_to(ROOT).as_posix()] = {
            'bytes': path.stat().st_size, 'sha256': digest}
    manifest_path = ROOT / 'CODE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive = ROOT / ARCHIVE_NAME
    with ZipFile(archive, 'w') as zipped:
        for path in [*files, manifest_path]:
            zipped.write(path, f'{TOP}/{path.relative_to(ROOT).as_posix()}',
                         compress_type=ZIP_STORED if path.suffix == '.zip' else ZIP_DEFLATED)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    (ROOT / f'{ARCHIVE_NAME}.sha256').write_text(f'{digest}  {ARCHIVE_NAME}\n', encoding='ascii')
    print(f'{archive}: {archive.stat().st_size:,} bytes; {len(files) + 1} files')


if __name__ == '__main__':
    main()
