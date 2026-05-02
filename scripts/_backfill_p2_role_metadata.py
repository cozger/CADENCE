"""Backfill p1_role / p2_role into existing P2 summary JSONs.

Reads each session's run log, extracts the p1=... line, and adds
'p1_role', 'p2_role', 'direction_p1_to_p2', 'direction_p2_to_p1' fields
to the corresponding _p2_summary.json. Idempotent — does not recompute
S-map; only adds metadata fields.

Without role metadata, downstream aggregators silently mismap cs_tp/cs_pt
to therapist/patient direction labels for any session where p1 is the
therapist instead of the patient.
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    ROOT / 'results' / 'native_rate_coupling_p2',
    ROOT / 'results' / 'native_rate_coupling_p2' / 'theta_grand_avg',
]

ROLE_LINE_RE = re.compile(r'p1\s*=\s*(\w+)\s*,\s*p2\s*=\s*(\w+)')


def find_run_log(json_path):
    """Try several conventions: {arg}_run.log next to JSON, or in either dir."""
    arg = json_path.stem.replace('_p2_summary', '')
    candidates = [
        json_path.parent / f'{arg}_run.log',
        ROOT / 'results' / 'native_rate_coupling_p2' / f'{arg}_run.log',
        ROOT / 'results' / 'native_rate_coupling_p2' / 'theta_grand_avg' / f'{arg}_run.log',
        ROOT / 'results' / 'native_rate_coupling_p2' / 'run_log.txt',  # y_06 used this name
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_role_from_log(log_path, session_arg):
    """Extract p1_role and p2_role from a run log. Returns (p1_role, p2_role)
    or (None, None) if not found."""
    text = None
    for enc in ('utf-8', 'utf-16-le', 'utf-16'):
        try:
            text = log_path.read_text(encoding=enc, errors='replace')
            if 'p1=' in text or 'p1 =' in text:
                break
        except Exception:
            continue
    if text is None:
        print(f'  Could not read log under any encoding: {log_path}')
        return None, None
    # Pick the first 'p1=...' line that follows a 'real=<session>' header
    # (run logs may concatenate multiple runs)
    real_marker = f'real={session_arg}'
    idx = text.find(real_marker)
    if idx < 0:
        idx = 0
    rest = text[idx:]
    m = ROLE_LINE_RE.search(rest)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def patch_json(json_path):
    arg = json_path.stem.replace('_p2_summary', '')
    with open(json_path) as f:
        data = json.load(f)
    if 'p1_role' in data and data['p1_role'] in ('patient', 'therapist'):
        print(f'[skip] {arg}: already has role metadata p1={data["p1_role"]}')
        return
    log_path = find_run_log(json_path)
    if log_path is None:
        print(f'[fail] {arg}: no run log found')
        return
    p1, p2 = parse_role_from_log(log_path, arg)
    if p1 is None:
        print(f'[fail] {arg}: could not parse role line in {log_path}')
        return
    data['p1_role'] = p1
    data['p2_role'] = p2
    data['direction_p1_to_p2'] = f'{p1} -> {p2}'
    data['direction_p2_to_p1'] = f'{p2} -> {p1}'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[ok  ] {arg}: p1={p1}, p2={p2}  ({log_path.name})')


def main():
    for dir_ in DIRS:
        if not dir_.exists():
            continue
        for json_path in sorted(dir_.glob('*_p2_summary.json')):
            patch_json(json_path)


if __name__ == '__main__':
    main()
