"""CADENCE Corpus-Wide EEG + BL Scan.

Runs multi-band amplitude coupling (EEG) and two-stage co-occurrence (BL)
on all sessions, per condition, role-resolved (therapist/patient).

All outputs in role terms, never p1/p2.
GPU-accelerated EEG via fast_cycles, parallelized BL surrogates.

Usage:
  python scripts/run_corpus_scan.py
  python scripts/run_corpus_scan.py --quick        # fewer surrogates
  python scripts/run_corpus_scan.py --eeg-only     # skip BL
  python scripts/run_corpus_scan.py --bl-only      # skip EEG
"""
import sys, os, time, json, argparse, warnings, re
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.conditions import parse_condition_intervals
from cadence.significance.fast_cycles import analyze_interbrain_cycles_multiband
from cadence.significance.bl_coupling import bl_two_stage_coupling

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
parser.add_argument('--eeg-only', action='store_true')
parser.add_argument('--bl-only', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output', default='results/corpus_scan.json')
args = parser.parse_args()

FS_EEG = 256.0; FS_BL = 30.0; N_CH = 14
K_EEG = 50 if args.quick else 200
K_BL = 100 if args.quick else 300
N_SURR_BL_EVENT = 100 if args.quick else 300

# ── Helpers ──────────────────────────────────────────────────────────────

def resolve_eeg_roles(s):
    p1r = s.get('p1_role', 'therapist')
    if p1r == 'therapist':
        return s['p1_eeg'], s['p2_eeg'], s['p1_eeg_ts'], s['p2_eeg_ts']
    return s['p2_eeg'], s['p1_eeg'], s['p2_eeg_ts'], s['p1_eeg_ts']


def resolve_bl_roles(s):
    p1r = s.get('p1_role', 'therapist')
    if p1r == 'therapist':
        return (s.get('p1_blendshapes'), s.get('p2_blendshapes'),
                s.get('p1_blendshapes_ts'), s.get('p2_blendshapes_ts'))
    return (s.get('p2_blendshapes'), s.get('p1_blendshapes'),
            s.get('p2_blendshapes_ts'), s.get('p1_blendshapes_ts'))


def prep_eeg(eeg, ts, start_s, end_s):
    m = (ts >= start_s) & (ts < end_s)
    N = int(m.sum())
    if N < int(10 * FS_EEG):
        return None, 0
    dur = end_s - start_s
    t = np.linspace(0, dur, N)
    seg = np.stack([np.interp(t, ts[m]-start_s, eeg[m, c])
                     for c in range(N_CH)], axis=1).astype(np.float64)
    seg -= seg.mean(axis=1, keepdims=True)
    for ch in range(N_CH):
        mu, sd = seg[:, ch].mean(), max(seg[:, ch].std(), 1e-8)
        seg[:, ch] = (seg[:, ch] - mu) / sd
    return seg.astype(np.float32), N


def prep_bl(bl, ts, start_s, end_s):
    if bl is None:
        return None, 0
    m = (ts >= start_s) & (ts < end_s)
    N = int(m.sum())
    n_au = min(bl.shape[1], 52)
    if N < int(10 * FS_BL):
        return None, 0
    dur = end_s - start_s
    t = np.linspace(0, dur, N)
    seg = np.stack([np.interp(t, ts[m]-start_s, bl[m, c])
                     for c in range(n_au)], axis=1).astype(np.float32)
    return seg, N


def dedup_sessions(entries):
    """Keep latest version of each session (by base name)."""
    by_name = {}
    for name, path in entries:
        # Strip hash prefix: "7a143771e6cb_y_06" → "y_06"
        base = re.sub(r'^[a-f0-9]+_', '', name)
        if base not in by_name or name > by_name[base][0]:
            by_name[base] = (name, path)
    return sorted(by_name.values())


# ── Load sessions ────────────────────────────────────────────────────────
cfg = load_config('configs/default.yaml')
entries = discover_cached_sessions(cfg['session_cache'])
unique = dedup_sessions(entries)

print(f"Found {len(entries)} cache entries, {len(unique)} unique sessions")
print(f"EEG surrogates: {K_EEG}, BL surrogates: {K_BL}")
print(f"Mode: {'EEG only' if args.eeg_only else 'BL only' if args.bl_only else 'EEG + BL'}\n")

sessions = []
for name, path in unique:
    s = load_session_from_cache(path, config=cfg)
    if s is not None and 'p1_eeg' in s:
        sessions.append((name, s))
        p1r = s.get('p1_role', '?')
        p2r = s.get('p2_role', '?')
        dur = s.get('duration', 0)
        intervals = parse_condition_intervals(s)
        conds = [c for _, _, c in intervals]
        print(f"  {name}: P1={p1r}, P2={p2r}, {dur:.0f}s, {conds}")

print(f"\n{len(sessions)} sessions loaded\n")

# ── Corpus scan ──────────────────────────────────────────────────────────
all_results = []

for sess_idx, (sess_name, session) in enumerate(sessions):
    intervals = parse_condition_intervals(session)
    p1_role = session.get('p1_role', '?')
    p2_role = session.get('p2_role', '?')
    t_eeg, p_eeg, t_ts, p_ts = resolve_eeg_roles(session)
    t_bl, p_bl, t_bl_ts, p_bl_ts = resolve_bl_roles(session)

    print(f"{'='*70}")
    print(f"  [{sess_idx+1}/{len(sessions)}] {sess_name} "
          f"(P1={p1_role}, P2={p2_role})")
    print(f"{'='*70}")

    for start, end, cond in intervals:
        dur = end - start
        if dur < 30:
            continue

        t0 = time.perf_counter()
        result = {
            'session': sess_name, 'condition': cond,
            'duration_s': dur, 'p1_role': p1_role, 'p2_role': p2_role,
        }

        # ── EEG ──────────────────────────────────────────────────────
        if not args.bl_only:
            t_seg, Nt = prep_eeg(t_eeg, t_ts, start, end)
            p_seg, Np = prep_eeg(p_eeg, p_ts, start, end)
            if t_seg is not None and p_seg is not None:
                N_eeg = min(Nt, Np)
                r_eeg = analyze_interbrain_cycles_multiband(
                    t_seg[:N_eeg], p_seg[:N_eeg], FS_EEG,
                    n_surrogates=K_EEG, seed=args.seed)

                for band in ['theta', 'alpha', 'beta']:
                    bz = r_eeg['per_band'][band]['volt_amp']['pooled_z']
                    result[f'eeg_{band}_z'] = round(bz, 2)
                cz = r_eeg['combined']['volt_amp']['stouffer_z']
                result['eeg_combined_z'] = round(cz, 2)

        # ── BL ───────────────────────────────────────────────────────
        if not args.eeg_only:
            t_bl_seg, _ = prep_bl(t_bl, t_bl_ts, start, end)
            p_bl_seg, _ = prep_bl(p_bl, p_bl_ts, start, end)
            if t_bl_seg is not None and p_bl_seg is not None:
                # Align lengths
                n_bl = min(len(t_bl_seg), len(p_bl_seg))
                t_bl_seg = t_bl_seg[:n_bl]
                p_bl_seg = p_bl_seg[:n_bl]
                for direction, src, tgt, label in [
                    ('T_to_P', t_bl_seg, p_bl_seg, 'T→P'),
                    ('P_to_T', p_bl_seg, t_bl_seg, 'P→T'),
                ]:
                    r_bl = bl_two_stage_coupling(
                        src, tgt, FS_BL,
                        max_lag_s=5.0, smooth_s=3.0,
                        n_surrogates=K_BL, target_fa=0.05,
                        event_prominence=0.3, min_event_iei_s=3.0,
                        n_surrogates_event=N_SURR_BL_EVENT,
                        population_prior=(2.5, 1.0), seed=args.seed)
                    cats = getattr(r_bl, 'catalogs', {})
                    for comp_name in ['smile', 'brow', 'general']:
                        cat = cats.get(comp_name)
                        if cat:
                            result[f'bl_{direction}_{comp_name}_p'] = round(
                                cat.session_p_value, 3)
                            result[f'bl_{direction}_{comp_name}_co'] = cat.n_cooccurrences

        elapsed = time.perf_counter() - t0
        result['elapsed_s'] = round(elapsed, 1)
        all_results.append(result)

        # Print row
        eeg_str = ""
        if 'eeg_combined_z' in result:
            eeg_str = (f"θ={result.get('eeg_theta_z',0):+.1f} "
                       f"α={result.get('eeg_alpha_z',0):+.1f} "
                       f"β={result.get('eeg_beta_z',0):+.1f} "
                       f"C={result.get('eeg_combined_z',0):+.1f}")
        bl_str = ""
        if f'bl_T_to_P_smile_p' in result:
            sp = result.get('bl_T_to_P_smile_p', 1)
            bp = result.get('bl_T_to_P_brow_p', 1)
            bl_str = (f"T→P sm={sp:.3f}{'*' if sp<0.05 else ' '} "
                      f"br={bp:.3f}{'*' if bp<0.05 else ' '}")

        print(f"  {cond:>15} ({dur:>4.0f}s, {elapsed:>4.1f}s) "
              f"{eeg_str}  {bl_str}")

# ── Save JSON ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {args.output}")

# ── Summary table ────────────────────────────────────────────────────────
print(f"\n\n{'='*90}")
print(f"  CORPUS SUMMARY: EEG amplitude coupling (volt_amp pooled z)")
print(f"{'='*90}")
print(f"  {'session':>20} {'condition':>15} {'dur':>5} "
      f"{'theta':>7} {'alpha':>7} {'beta':>7} {'combined':>9}")
print(f"  {'-'*20} {'-'*15} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")

for r in sorted(all_results, key=lambda x: (x['session'], x['condition'])):
    if 'eeg_combined_z' not in r:
        continue
    print(f"  {r['session']:>20} {r['condition']:>15} {r['duration_s']:>4.0f}s "
          f"{r.get('eeg_theta_z',0):>+6.1f} {r.get('eeg_alpha_z',0):>+6.1f} "
          f"{r.get('eeg_beta_z',0):>+6.1f} {r.get('eeg_combined_z',0):>+8.1f}")

# Per-condition averages
print(f"\n  {'CONDITION AVERAGES':>20}")
cond_groups = {}
for r in all_results:
    c = r['condition']
    if 'eeg_combined_z' in r:
        cond_groups.setdefault(c, []).append(r)

for cond in sorted(cond_groups.keys()):
    rs = cond_groups[cond]
    n = len(rs)
    tz = np.mean([r.get('eeg_theta_z', 0) for r in rs])
    az = np.mean([r.get('eeg_alpha_z', 0) for r in rs])
    bz = np.mean([r.get('eeg_beta_z', 0) for r in rs])
    cz = np.mean([r.get('eeg_combined_z', 0) for r in rs])
    print(f"  {'avg('+str(n)+')':>20} {cond:>15}       "
          f"{tz:>+6.1f} {az:>+6.1f} {bz:>+6.1f} {cz:>+8.1f}")

# BL summary
if not args.eeg_only:
    print(f"\n{'='*90}")
    print(f"  CORPUS SUMMARY: BL co-occurrence (T→P smile/brow p-values)")
    print(f"{'='*90}")
    for r in sorted(all_results, key=lambda x: (x['session'], x['condition'])):
        sp = r.get('bl_T_to_P_smile_p')
        bp = r.get('bl_T_to_P_brow_p')
        if sp is None:
            continue
        print(f"  {r['session']:>20} {r['condition']:>15} "
              f"smile={sp:.3f}{'*' if sp<0.05 else ' '} "
              f"brow={bp:.3f}{'*' if bp<0.05 else ' '}")

print(f"\n{'='*90}")
total_time = sum(r.get('elapsed_s', 0) for r in all_results)
print(f"  Total: {len(all_results)} condition-segments, {total_time:.0f}s")
print(f"{'='*90}")
