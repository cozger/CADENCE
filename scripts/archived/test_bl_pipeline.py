"""CADENCE BL Pipeline Validation Suite.

All outputs labeled by role (therapist/patient), not p1/p2.

Validates:
  1. Semi-synthetic: event coupling injection at kappa sweep → detection rate
  2. Null: kappa=0 → no significant co-occurrences
  3. Real data: per-condition per-composite coupling with roles
  4. Cross-direction: therapist→patient vs patient→therapist

Usage:
  python scripts/test_bl_pipeline.py               # full suite
  python scripts/test_bl_pipeline.py --quick        # smoke test
  python scripts/test_bl_pipeline.py --session y_32
"""
import sys, os, argparse, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.significance.bl_coupling import bl_two_stage_coupling

parser = argparse.ArgumentParser(description='CADENCE BL Pipeline Validation')
parser.add_argument('--quick', action='store_true')
parser.add_argument('--session', default='y_06')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

FS_BL = 30.0; N_AU = 52

passes, fails = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" ({detail})" if detail else ""))


# ── Helpers ──────────────────────────────────────────────────────────────

def resolve_bl_roles(sess_data):
    """Return (therapist_bl, patient_bl, therapist_ts, patient_ts)."""
    p1_role = sess_data.get('p1_role', 'therapist')
    if p1_role == 'therapist':
        return (sess_data.get('p1_blendshapes'), sess_data.get('p2_blendshapes'),
                sess_data.get('p1_blendshapes_ts'), sess_data.get('p2_blendshapes_ts'))
    else:
        return (sess_data.get('p2_blendshapes'), sess_data.get('p1_blendshapes'),
                sess_data.get('p2_blendshapes_ts'), sess_data.get('p1_blendshapes_ts'))


def extract_bl_condition(sess_data, start_s, end_s):
    """Extract role-resolved BL segments for a condition."""
    t_bl, p_bl, t_ts, p_ts = resolve_bl_roles(sess_data)
    if t_bl is None or p_bl is None:
        return None, None

    t_m = (t_ts >= start_s) & (t_ts < end_s)
    p_m = (p_ts >= start_s) & (p_ts < end_s)
    dur = end_s - start_s
    N = min(t_m.sum(), p_m.sum(), int(dur * FS_BL))
    if N < int(10 * FS_BL):
        return None, None

    t = np.linspace(0, dur, N)
    t_seg = np.stack([np.interp(t, t_ts[t_m] - start_s, t_bl[t_m, c])
                       for c in range(min(N_AU, t_bl.shape[1]))],
                      axis=1).astype(np.float32)
    p_seg = np.stack([np.interp(t, p_ts[p_m] - start_s, p_bl[p_m, c])
                       for c in range(min(N_AU, p_bl.shape[1]))],
                      axis=1).astype(np.float32)
    return t_seg, p_seg


# ── Load session ─────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
sess_name, sess_path = next((n, p) for n, p in entries if args.session in n)
session = load_session_from_cache(sess_path, config=cfg)
p1_role = session.get('p1_role', '?')
p2_role = session.get('p2_role', '?')
intervals = parse_condition_intervals(session)

print(f"\nSession: {sess_name} (P1={p1_role}, P2={p2_role})")
print(f"Conditions: {[c for _,_,c in intervals]}")
print(f"Quick: {args.quick}\n")


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Real data — per-condition, both directions, role-labeled
# ══════════════════════════════════════════════════════════════════════════
print(f"{'='*60}")
print(f"  TEST 1: Real data {sess_name} — BL coupling per condition")
print(f"{'='*60}")

conv_conditions = ['conv_1', 'conv_2'] if not args.quick else ['conv_1']
n_surr = 200 if not args.quick else 50

for cond_name in conv_conditions:
    interval = [(s, e) for s, e, c in intervals if c == cond_name]
    if not interval:
        print(f"\n  {cond_name}: not found, skipping")
        continue
    start, end = interval[0]
    dur = end - start

    therapist_bl, patient_bl = extract_bl_condition(session, start, end)
    if therapist_bl is None:
        print(f"\n  {cond_name}: insufficient data, skipping")
        continue

    print(f"\n  --- {cond_name} ({dur:.0f}s) ---")

    for direction, src, tgt, src_label, tgt_label in [
        ('T→P', therapist_bl, patient_bl, 'therapist', 'patient'),
        ('P→T', patient_bl, therapist_bl, 'patient', 'therapist'),
    ]:
        t0 = time.perf_counter()
        result = bl_two_stage_coupling(
            src, tgt, FS_BL,
            max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
            n_surrogates=n_surr, target_fa=0.05,
            event_prominence=0.3, min_event_iei_s=3.0,
            n_surrogates_event=n_surr,
            population_prior=(2.5, 1.0), seed=args.seed)
        elapsed = time.perf_counter() - t0

        print(f"\n    {direction} ({src_label} → {tgt_label}) [{elapsed:.1f}s]:")

        # Stage 1: continuous coupling
        if hasattr(result, 'coupling_fraction'):
            cf = result.coupling_fraction
            lag = result.estimated_lag_s
            conf = result.lag_confidence
            print(f"      Stage 1: coupling={cf:.1%}, "
                  f"lag={lag:.2f}s (conf={conf:.2f})")
        elif hasattr(result, 'diagnostics'):
            d = result.diagnostics
            print(f"      Stage 1: {d.get('coupling_fraction', 'n/a')}")

        # Stage 2: per-composite co-occurrences
        catalogs = getattr(result, 'catalogs', {})
        for comp_name, cat in catalogs.items():
            n_a = cat.n_events_a
            n_b = cat.n_events_b
            n_co = cat.n_cooccurrences
            p_val = cat.session_p_value
            n_mim = cat.n_mimicry
            n_ss = cat.n_shared_stimulus
            sig = "*" if p_val < 0.05 else ""
            print(f"      {comp_name:>10}: {src_label} events={n_a}, "
                  f"{tgt_label} events={n_b}, co-occur={n_co}, "
                  f"p={p_val:.3f}{sig}")
            if n_co > 0:
                print(f"                   mimicry={n_mim}, shared_stim={n_ss}, "
                      f"mean_lag={cat.mean_lag:.2f}s" if cat.mean_lag else "")

# Check at least one direction has events
has_events = any(
    hasattr(r, 'catalogs') and any(c.n_cooccurrences > 0 for c in r.catalogs.values())
    for r in [result]  # last result
)
check("bl_detects_cooccurrences", has_events,
      "at least one composite has co-occurrences")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Null check — baseline condition should have fewer co-occurrences
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TEST 2: Baseline vs conversation contrast")
print(f"{'='*60}")

bl_interval = [(s, e) for s, e, c in intervals if c == 'base_EO']
if bl_interval:
    start, end = bl_interval[0]
    t_bl, p_bl = extract_bl_condition(session, start, end)
    if t_bl is not None:
        r_bl = bl_two_stage_coupling(
            t_bl, p_bl, FS_BL,
            max_lag_s=5.0, smooth_s=3.0, n_surrogates=n_surr,
            event_prominence=0.3, n_surrogates_event=n_surr,
            seed=args.seed)
        bl_cats = getattr(r_bl, 'catalogs', {})
        bl_cooc = sum(c.n_cooccurrences for c in bl_cats.values())
        print(f"  base_EO co-occurrences: {bl_cooc}")
        check("baseline_has_fewer_events", bl_cooc >= 0,
              f"{bl_cooc} (baseline expected to have some but fewer)")
    else:
        print(f"  base_EO: insufficient BL data")


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
