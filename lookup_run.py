#!/usr/bin/env python3
"""Find a published compiler by run ID, path, old branch, tag or source hash."""
import json
from pathlib import Path
import sys

rows = json.loads(Path(__file__).with_name('runs.json').read_text())['snapshots']
query = ' '.join(sys.argv[1:]).strip()
if not query:
    raise SystemExit('Usage: python3 lookup_run.py RUN_ID_OR_PATH_OR_SHA')
def keys(row):
    return row['aliases'] + [row['tag'], row['branch'] or '', row['display_name']]
exact = [r for r in rows if query in keys(r)]
matches = exact or [r for r in rows if any(query.lower() in k.lower() for k in keys(r))]
if not matches:
    raise SystemExit('No matching run. See runs.csv for the complete inventory.')
for row in matches:
    print(json.dumps({k: row.get(k) for k in ['label', 'display_name', 'branch', 'tag',
          'commit', 'branch_commit', 'source_sha256', 'snapshot_url', 'branch_url',
          'recorded_study_role']}, indent=2))
