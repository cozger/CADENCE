"""CADENCE EEG Pipeline Validation Suite.

All outputs labeled by role (therapist/patient), not p1/p2.

Validates:
  1. Semi-synthetic: arousal injection kappa sweep → verify selectivity
  2. PLV cross-check: arousal → PLV should NOT detect
  3. Real data: per-condition multi-band scan with roles
  4. Pseudo-dyad null: cross-session → z≈0

Usage:
  python scripts/test_eeg_pipeline.py              # full suite
  python scripts/test_eeg_pipeline.py --quick       # smoke test (~30s)
  python scripts/test_eeg_pipeline.py --session y_32
"""
import sys, os, argparse, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.synthetic import (generate_coupling_gate, find_valid_window,
                                inject_eeg_coupling_arousal)
from cadence.significance.fast_cycles import (
    analyze_interbrain_cycles_multiband, EEG_BANDS)
from cadence.significance.coherence_localization import (
    wpli_temporal_localization, _min_event_filter)

parser = argparse.ArgumentParser(description='CADENCE EEG Pipeline Validation')
parser.add_argument('--quick', action='store_true')
parser.add_argument('--session', default='y_06')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

FS = 256.0; N_CH = 14
K_SURR = 50 if args.quick else 200
KAPPAS = [0.30] if args.quick else [0.0, 0.15, 0.30, 0.50]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THRESHOLDS = {
    'null_z': 2.0,
    'detect_z': 2.0,
    'selectivity_z': 3.0,
    'pseudo_null_z': 2.0,
    'plv_no_detect_pct': 15.0,
}

passes, fails = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" ({detail})" if detail else ""))


# ── Helpers ──────────────────────────────────────────────────────────────

def resolve_roles(sess_data):
    """Return (therapist_eeg, patient_eeg, therapist_ts, patient_ts)."""
    p1_role = sess_data.get('p1_role', 'therapist')
    if p1_role == 'therapist':
        return (sess_data['p1_eeg'], sess_data['p2_eeg'],
                sess_data['p1_eeg_ts'], sess_data['p2_eeg_ts'])
    else:
        return (sess_data['p2_eeg'], sess_data['p1_eeg'],
                sess_data['p2_eeg_ts'], sess_data['p1_eeg_ts'])


def prep_segment(eeg, ts, start_s, end_s):
    """Extract, avg-ref, z-score one participant's EEG segment."""
    m = (ts >= start_s) & (ts < end_s)
    N = int(m.sum())
    if N < int(10 * FS):
        return None, 0
    dur = end_s - start_s
    t = np.linspace(0, dur, N)
    seg = np.stack([np.interp(t, ts[m] - start_s, eeg[m, c])
                     for c in range(N_CH)], axis=1).astype(np.float64)
    seg -= seg.mean(axis=1, keepdims=True)  # avg ref
    for ch in range(N_CH):
        mu, sd = seg[:, ch].mean(), max(seg[:, ch].std(), 1e-8)
        seg[:, ch] = (seg[:, ch] - mu) / sd
    return seg.astype(np.float32), N


def extract_dyad(sess_data, start_s, end_s):
    """Extract role-resolved (therapist, patient) EEG pair."""
    t_eeg, p_eeg, t_ts, p_ts = resolve_roles(sess_data)
    t_seg, Nt = prep_segment(t_eeg, t_ts, start_s, end_s)
    p_seg, Np = prep_segment(p_eeg, p_ts, start_s, end_s)
    if t_seg is None or p_seg is None:
        return None, None
    N = min(Nt, Np)
    return t_seg[:N], p_seg[:N]


# ── Load sessions ────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])

sess_name, sess_path = next((n, p) for n, p in entries if args.session in n)
session = load_session_from_cache(sess_path, config=cfg)
p1_role = session.get('p1_role', '?')
p2_role = session.get('p2_role', '?')
intervals = parse_condition_intervals(session)

# Another session for cross-dyad semi-synthetic + pseudo-null
all_sess = [(n, load_session_from_cache(p, cfg)) for n, p in entries]
all_sess = [(n, s) for n, s in all_sess if s is not None and 'p1_eeg' in s]
s_other = next(s for n, s in all_sess if args.session not in n)

print(f"\nSession: {sess_name} (P1={p1_role}, P2={p2_role})")
print(f"Config: K={K_SURR}, device={device}, quick={args.quick}")
print(f"Conditions: {[c for _,_,c in intervals]}\n")


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Semi-synthetic arousal kappa sweep
# ══════════════════════════════════════════════════════════════════════════
print(f"{'='*60}")
print(f"  TEST 1: Semi-synthetic arousal injection")
print(f"  (cross-dyad: therapist EEG from {sess_name},"
      f" patient EEG from different session)")
print(f"{'='*60}")

# Cross-dyad: therapist from primary session, patient from other session
t_eeg_raw, _, t_ts, _ = resolve_roles(session)
_, p_eeg_raw, _, p_ts = resolve_roles(s_other)

# Use full session range (from timestamps)
t_start, t_end = float(t_ts[0]), min(float(t_ts[-1]), float(t_ts[0]) + 1800)
p_start, p_end = float(p_ts[0]), min(float(p_ts[-1]), float(p_ts[0]) + 1800)
dur_ss = min(t_end - t_start, p_end - p_start)

therapist_ss, Nt = prep_segment(t_eeg_raw, t_ts, t_start, t_start + dur_ss)
patient_ss, Np = prep_segment(p_eeg_raw, p_ts, p_start, p_start + dur_ss)
N_ss = min(Nt, Np)
therapist_ss = therapist_ss[:N_ss]
patient_ss = patient_ss[:N_ss]

gate = generate_coupling_gate(N_ss, FS, {
    'duty_cycle': 0.10, 'event_range_s': (5, 20), 'ramp_s': 1.0}, seed=args.seed)
t_eeg_grid = np.arange(N_ss) / FS

print(f"\n  Injection: therapist arousal → patient amplitude modulation")
print(f"  {'kappa':>6} {'theta':>8} {'alpha':>8} {'beta':>8} "
      f"{'period':>8} {'symmetry':>9}")

for kappa in KAPPAS:
    if kappa == 0:
        p2c = patient_ss.copy()
    else:
        p2c, _, _ = inject_eeg_coupling_arousal(
            therapist_ss, patient_ss, gate, kappa,
            lag_s=1.0, source_band=(4.0, 8.0),
            spatial_mode='frontal', fs=FS, seed=args.seed)
        for ch in range(N_CH):
            mu, sd = p2c[:, ch].mean(), max(p2c[:, ch].std(), 1e-8)
            p2c[:, ch] = (p2c[:, ch] - mu) / sd

    r = analyze_interbrain_cycles_multiband(
        therapist_ss, p2c, FS, n_surrogates=K_SURR, seed=args.seed)

    tz = r['per_band']['theta']['volt_amp']['pooled_z']
    az = r['per_band']['alpha']['volt_amp']['pooled_z']
    bz = r['per_band']['beta']['volt_amp']['pooled_z']
    tp = r['per_band']['theta']['period']['pooled_z']
    ts = r['per_band']['theta']['time_rdsym']['pooled_z']

    print(f"  {kappa:>6.2f} {tz:>+7.2f} {az:>+7.2f} {bz:>+7.2f} "
          f"{tp:>+7.2f} {ts:>+8.2f}")

    if kappa == 0:
        check("null_volt_amp_near_zero", abs(tz) < THRESHOLDS['null_z'],
              f"theta z={tz:+.2f}")
    elif kappa == 0.30:
        check("arousal_detected_theta", tz > THRESHOLDS['detect_z'],
              f"theta z={tz:+.2f}")
        check("period_not_detected", abs(tp) < THRESHOLDS['selectivity_z'],
              f"period z={tp:+.2f}")
        check("symmetry_not_detected", abs(ts) < THRESHOLDS['selectivity_z'],
              f"symmetry z={ts:+.2f}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: PLV cross-check
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TEST 2: PLV cross-check (kappa=0.30 arousal → PLV?)")
print(f"{'='*60}")

p2_plv, _, _ = inject_eeg_coupling_arousal(
    therapist_ss, patient_ss, gate, 0.30,
    lag_s=1.0, source_band=(4.0, 8.0),
    spatial_mode='frontal', fs=FS, seed=args.seed)
for ch in range(N_CH):
    mu, sd = p2_plv[:, ch].mean(), max(p2_plv[:, ch].std(), 1e-8)
    p2_plv[:, ch] = (p2_plv[:, ch] - mu) / sd

freqs_theta = np.logspace(np.log10(4.0), np.log10(8.0), 5)
_, z_plv, _, d_plv = wpli_temporal_localization(
    therapist_ss, p2_plv, FS, channels=list(range(N_CH)),
    center_freqs=freqs_theta, n_surrogates=min(K_SURR, 50),
    n_cycles=[3, 7], window_s=20.0, stride_s=0.5, smooth_s=15.0,
    target_fa=0.03, min_event_s=5.0, metric='plv',
    seed=args.seed, device=device)
gw = np.interp(d_plv['win_times'], t_eeg_grid, gate.astype(float)) > 0.5
plv_hit = 0
for thr in np.arange(0.4, 3.0, 0.1):
    m = _min_event_filter(z_plv > thr, 10)
    nc, nn = gw.sum(), (~gw).sum()
    h = float((m & gw).sum() / max(nc, 1))
    f = float((m & ~gw).sum() / max(nn, 1))
    if 0.03 <= f <= 0.07:
        plv_hit = max(plv_hit, h)

print(f"  PLV hit at FA~5%: {plv_hit:.1%}")
check("plv_does_not_detect_arousal",
      plv_hit * 100 < THRESHOLDS['plv_no_detect_pct'],
      f"hit={plv_hit:.1%}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Real data — per-condition, role-labeled
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TEST 3: Real data {sess_name}")
print(f"  Therapist = {session.get('p1_role','?')=='therapist' and 'P1' or 'P2'}"
      f", Patient = {session.get('p1_role','?')=='patient' and 'P1' or 'P2'}")
print(f"{'='*60}")

real_results = {}
print(f"\n  {'condition':>15} {'theta':>8} {'alpha':>8} {'beta':>8} {'combined':>10}")

for start, end, cond in intervals:
    therapist_seg, patient_seg = extract_dyad(session, start, end)
    if therapist_seg is None:
        continue

    r = analyze_interbrain_cycles_multiband(
        therapist_seg, patient_seg, FS,
        n_surrogates=K_SURR, seed=args.seed)
    real_results[cond] = r

    tz = r['per_band']['theta']['volt_amp']['pooled_z']
    az = r['per_band']['alpha']['volt_amp']['pooled_z']
    bz = r['per_band']['beta']['volt_amp']['pooled_z']
    cz = r['combined']['volt_amp']['stouffer_z']
    print(f"  {cond:>15} {tz:>+7.2f} {az:>+7.2f} {bz:>+7.2f} {cz:>+9.2f}")

social = [c for c in ['conv_1', 'conv_2', 'meditate_K'] if c in real_results]
baseline = [c for c in ['base_EO', 'base_EC'] if c in real_results]

if social and baseline:
    best_soc = max(real_results[c]['combined']['volt_amp']['stouffer_z'] for c in social)
    best_bl = max(real_results[c]['combined']['volt_amp']['stouffer_z'] for c in baseline)
    check("social_stronger_than_baseline",
          best_soc > best_bl,
          f"social z={best_soc:+.1f} > baseline z={best_bl:+.1f}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Pseudo-dyad null (therapist from session, patient from other)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TEST 4: Pseudo-dyad null (cross-session)")
print(f"{'='*60}")

longest = max(intervals, key=lambda x: x[1] - x[0])
t_eeg_real, _, t_ts_real, _ = resolve_roles(session)
_, p_eeg_other, _, p_ts_other = resolve_roles(s_other)

therapist_null, Nt = prep_segment(t_eeg_real, t_ts_real, longest[0], longest[1])
dur_null = longest[1] - longest[0]
p_start_o = float(p_ts_other[0])
patient_null, Np = prep_segment(p_eeg_other, p_ts_other, p_start_o, p_start_o + dur_null)
N_null = min(Nt, Np)

r_null = analyze_interbrain_cycles_multiband(
    therapist_null[:N_null], patient_null[:N_null], FS,
    n_surrogates=K_SURR, seed=args.seed)

null_cz = r_null['combined']['volt_amp']['stouffer_z']
print(f"  Combined volt_amp z: {null_cz:+.2f}")
check("pseudo_dyad_null_clean",
      abs(null_cz) < THRESHOLDS['pseudo_null_z'],
      f"z={null_cz:+.2f}")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
n_pass, n_fail = len(passes), len(fails)
print(f"  RESULTS: {n_pass} passed, {n_fail} failed")
print(f"{'='*60}")
for p in passes:
    print(f"  [PASS] {p}")
for f in fails:
    print(f"  [FAIL] {f}")
status = "PASS" if n_fail == 0 else "FAIL"
print(f"\n  Overall: {status}")
print(f"{'='*60}")
sys.exit(0 if n_fail == 0 else 1)
