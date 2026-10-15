"""Download and SHA-256-check the recorded experiment data using the standard library."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    record = json.loads((ROOT / 'data/manifest.json').read_text())['data_archive']
    target = ROOT / 'data' / record['filename']
    if target.exists():
        if digest(target) != record['sha256']:
            raise RuntimeError('Existing data has a different checksum; preserve it and choose a clean checkout.')
        print(f'Already verified: {target.name}')
        return
    partial = target.with_suffix('.zip.part')
    if partial.exists():
        raise RuntimeError('A partial download exists; inspect or remove it before retrying.')
    request = urllib.request.Request(record['download_url'], headers={'User-Agent': 'crn-policy-data-download'})
    done = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as out:
        while block := response.read(4 * 1024 * 1024):
            out.write(block)
            done += len(block)
            print(f'\rDownloaded {done / 1e6:.1f} / {record["bytes"] / 1e6:.1f} MB', end='', flush=True)
    print()
    if done != record['bytes'] or digest(partial) != record['sha256']:
        raise RuntimeError('Download verification failed; the partial file was retained for inspection.')
    partial.rename(target)
    print(f'Verified: {target.name}')


if __name__ == '__main__':
    main()
