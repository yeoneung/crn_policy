"""Build and package the flat Automatica LaTeX source in an isolated directory."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ['main.tex', 'ref.bib', 'autart.cls', 'elsarticle-harv.bst',
         'certification_mechanism.pdf']


def tex_command(name):
    found = shutil.which(name)
    if found:
        return found
    directories = ['/Library/TeX/texbin',
                   '/tmp/mcs-tex/TinyTeX/bin/universal-darwin']
    if os.environ.get('AUTOMATICA_TEXBIN'):
        directories.insert(0, os.environ['AUTOMATICA_TEXBIN'])
    for directory in directories:
        candidate = Path(directory) / name
        if candidate.exists():
            return str(candidate)
    raise RuntimeError('Install a TeX distribution, or set AUTOMATICA_TEXBIN to its bin directory.')


def compile_at(directory):
    env = dict(os.environ, LC_ALL='C', LANG='C')
    env['PATH'] = str(Path(tex_command('pdflatex')).parent) + os.pathsep + env['PATH']
    commands = [
        [tex_command('pdflatex'), '-interaction=nonstopmode', '-halt-on-error',
         '-file-line-error', 'main.tex'],
        [tex_command('bibtex'), 'main'],
        [tex_command('pdflatex'), '-interaction=nonstopmode', '-halt-on-error',
         '-file-line-error', 'main.tex'],
        [tex_command('pdflatex'), '-interaction=nonstopmode', '-halt-on-error',
         '-file-line-error', 'main.tex'],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=directory, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            (ROOT / 'output').mkdir(exist_ok=True)
            (ROOT / 'output/failed_build.log').write_bytes(result.stdout)
            raise RuntimeError(result.stdout.decode(errors='replace')[-6500:])
    log = (directory / 'main.log').read_text(errors='replace')
    # Tables with forward references can require another pass after BibTeX.
    for _ in range(3):
        if 'Rerun to get' not in log and 'Label(s) may have changed' not in log:
            break
        result = subprocess.run(commands[-1], cwd=directory, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout.decode(errors='replace')[-6500:])
        log = (directory / 'main.log').read_text(errors='replace')
    warnings = re.findall(r'^.*(?:undefined|multiply defined|same identifier|Overfull|Font Warning).*$',
                          log, re.MULTILINE)
    if 'Rerun to get' in log or 'Label(s) may have changed' in log:
        warnings.append('Cross-references did not converge.')
    if warnings:
        raise RuntimeError('\n'.join(warnings))
    return log


def main():
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='automatica-build-') as name:
        directory = Path(name)
        for item in FILES:
            shutil.copy2(ROOT / item, directory / item)
        log = compile_at(directory)
        shutil.copy2(directory / 'main.pdf', ROOT / 'main.pdf')
        shutil.copy2(directory / 'main.pdf', output / 'Automatica_manuscript.pdf')
        shutil.copy2(directory / 'main.bbl', ROOT / 'main.bbl')
        shutil.copy2(directory / 'main.log', output / 'build.log')
        shutil.copy2(directory / 'main.blg', output / 'bibliography.log')
        archive = output / 'Automatica_source.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for item in FILES + ['main.bbl']:
                zipped.write(directory / item, item)
        # The submission source must build without its parent folder or artifact.
        with tempfile.TemporaryDirectory(prefix='automatica-source-check-') as check:
            check_path = Path(check)
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(check_path)
            compile_at(check_path)
            from pypdf import PdfReader
            original = PdfReader(directory / 'main.pdf')
            rebuilt = PdfReader(check_path / 'main.pdf')
            assert [p.extract_text() for p in original.pages] == [p.extract_text() for p in rebuilt.pages]
            pages = len(original.pages)
        source = (ROOT / 'main.tex').read_text()
        citations = set()
        for keys in re.findall(r'\\cite\w*\s*(?:\[[^\]]*\]\s*)?\{([^}]*)\}', source):
            citations.update(k.strip() for k in keys.split(','))
        bibkeys = set(re.findall(r'@\w+\{([^,]+),', (ROOT / 'ref.bib').read_text()))
        assert citations == bibkeys, (citations - bibkeys, bibkeys - citations)
        labels = re.findall(r'\\label\{([^}]+)\}', source)
        references = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', source)
        assert len(labels) == len(set(labels))
        assert set(references) <= set(labels)
        assert not re.search(r'\\(?:input|include)\s*\{', source)
        report = {'pages_two_column': pages, 'cited_references': len(citations),
                  'single_main_tex': True, 'flat_source_archive': True,
                  'isolated_source_rebuild_matches_text': True,
                  'unresolved_references': False, 'overfull_boxes': False,
                  'font_substitution_warnings': False,
                  'files': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                            for f in FILES + ['main.bbl']}}
        (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Built {pages} two-column pages; {len(citations)} references; isolated source rebuild passed.')


if __name__ == '__main__':
    main()
