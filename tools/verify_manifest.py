"""Verify SHA-256 and size of all archived sources/data, without rerunning experiments."""
from pathlib import Path
import hashlib
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / 'MANIFEST_SHA256.csv'
issues = []
with manifest.open(newline='', encoding='utf-8') as stream:
    records = list(csv.DictReader(stream))
for record in records:
    p = ROOT / record['relative_path']
    if not p.is_file():
        issues.append('MISSING: ' + record['relative_path'])
        continue
    if str(p.stat().st_size) != record['size_bytes']:
        issues.append('SIZE: ' + record['relative_path'])
    if hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
        issues.append('HASH: ' + record['relative_path'])
def _tracked(p):
    rel = p.relative_to(ROOT).parts
    # ignore VCS metadata and common local-only artefacts (see .gitignore)
    skip = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.venv'}
    return (p.is_file() and p.name != manifest.name
            and not any(part in skip for part in rel)
            and not p.name.endswith(('.pyc', '.pyo'))
            and p.name not in {'.DS_Store', 'Thumbs.db'})

current = {str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*') if _tracked(p)}
listed = {r['relative_path'] for r in records}
for extra in sorted(current - listed):
    issues.append('UNLISTED: ' + extra)
print(f"SHA-256 manifest: {len(records)} records; {len(issues)} issues")
for issue in issues:
    print(issue)
sys.exit(bool(issues))
