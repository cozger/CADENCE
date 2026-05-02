"""Batch theta-only S-map on all sessions with available XDFs.

Sessions limited to those with XDFs in raw sessions/ (8 of 11 protocol sessions):
    Meditation: y_06, y_17, y_19_3242026
    PE:         y01_021726, y05_02192026, Y_10_03182026, y_32_03132026, Y_41_03192026

Reuses existing y_06_p2_summary.json + y_32_p2_summary.json (both contain theta).
Outputs per-session theta JSON in results/native_rate_coupling_p2/theta_grand_avg/.
"""
import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, 'results', 'native_rate_coupling_p2', 'theta_grand_avg')
os.makedirs(OUT_DIR, exist_ok=True)

PYTHON_EXE = r"C:\Users\optilab\miniconda3\envs\MCCT\python.exe"
ENV_PATH = (r"C:\Users\optilab\miniconda3\envs\MCCT;"
            r"C:\Users\optilab\miniconda3\envs\MCCT\Library\bin;"
            r"C:\Users\optilab\miniconda3\envs\MCCT\Scripts;")

SESSIONS = [
    # (display_name, --session arg, protocol)
    ('y_06',           'y_06',            'meditation'),
    ('y_17',           'y_17',            'meditation'),
    ('y_19',           'y_19_3242026',    'meditation'),  # cache lacks EEG, will fail
    ('y_04',           'y04_020626',      'meditation'),
    ('y_11',           'y11_022526',      'meditation'),
    ('y_24',           'y24_022526',      'meditation'),
    ('y_01',           'y01_021726',      'PE'),
    ('y_05',           'y05_02192026',    'PE'),
    ('y_10',           'Y_10_03182026',   'PE'),
    ('y_32',           'y_32_03132026',   'PE'),
    ('y_41',           'Y_41_03192026',   'PE'),
]


def run_session(session_arg):
    """Run P2 script with theta-only, no ECG, no pseudo. Save to OUT_DIR."""
    out_json = os.path.join(OUT_DIR, f'{session_arg}_p2_summary.json')
    if os.path.exists(out_json):
        print(f'[SKIP] {session_arg}: output exists at {out_json}')
        return True
    print(f'[RUN ] {session_arg}: theta-only S-map ...', flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env['Path'] = ENV_PATH + env.get('Path', '')
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    env['PYTHONIOENCODING'] = 'utf-8'
    cmd = [
        PYTHON_EXE,
        os.path.join(ROOT, 'scripts', '_test_native_rate_coupling_p2.py'),
        '--session', session_arg,
        '--skip-pseudo',
        '--n-surr', '25',
        '--bands-eeg', 'theta',
        '--bands-ecg', '',
        '--out-dir', OUT_DIR,
    ]
    log_path = os.path.join(OUT_DIR, f'{session_arg}_run.log')
    with open(log_path, 'w', encoding='utf-8') as logf:
        result = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    ok = (result.returncode == 0) and os.path.exists(out_json)
    print(f'       {"OK " if ok else "FAIL"}  {elapsed:.0f}s  log={log_path}')
    return ok


def main():
    print(f'OUT_DIR = {OUT_DIR}')
    print(f'sessions: {len(SESSIONS)}')
    print()
    t0 = time.time()
    n_ok = 0
    for display, arg, protocol in SESSIONS:
        print(f'=== {display} ({protocol}) ===')
        if run_session(arg):
            n_ok += 1
    print()
    print(f'Done: {n_ok}/{len(SESSIONS)} sessions, total {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
