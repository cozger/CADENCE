"""Empirical emission profile for OTHER state during baseline conditions.

Loads each session's raw z-timecourses + gamma posteriors from rSLDS,
computes condition-weighted mean z-value per channel conditional on
being in the OTHER state during base_EO / base_EC.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cadence.constants import V11_MODALITY_KEYS

RESULTS_DIR = 'results/v11'

# Load hierarchical results to get state labels
hier = json.load(open(os.path.join(RESULTS_DIR, 'hierarchical', 'v11_hierarchical_results.json')))
state_labels = hier['state_labels']
other_idx = state_labels.index('OTHER')
coup_idx = state_labels.index('COUP')
shared_idx = state_labels.index('SHARED')
null_idx = state_labels.index('NULL')

print(f"State labels: {state_labels}")
print(f"OTHER = state {other_idx}")

# Collect per-session contributions
# For each channel d, for each condition c: sum(gamma[t, OTHER] * z_raw[t, d])
#                                         / sum(gamma[t, OTHER])
CONDITIONS = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K',
              'PE_1', 'PE_2', 'conv_2']

# For comparison, also compute means for COUP and SHARED states
state_names = ['OTHER', 'COUP', 'SHARED']
state_idxs = [other_idx, coup_idx, shared_idx]

# Accumulators: state -> condition -> channel -> list of weighted means (one per session)
acc = {s: {c: {k: [] for k in V11_MODALITY_KEYS} for c in CONDITIONS} for s in state_names}
# Also track unconditional baseline means (for reference)
uncond = {c: {k: [] for k in V11_MODALITY_KEYS} for c in CONDITIONS}

n_sessions = 0
for sess_dir in sorted(os.listdir(RESULTS_DIR)):
    full = os.path.join(RESULTS_DIR, sess_dir)
    if not os.path.isdir(full):
        continue
    scaffold_npz = os.path.join(full, 'scaffold_v11_ztimecourses.npz')
    rslds_npz = os.path.join(full, 'v11_rslds_results.npz')
    scaffold_json = os.path.join(full, 'scaffold_v11_results.json')
    if not all(os.path.exists(p) for p in [scaffold_npz, rslds_npz, scaffold_json]):
        continue

    npz = np.load(scaffold_npz)
    rslds = np.load(rslds_npz)
    meta = json.load(open(scaffold_json))

    t_common = npz['t_common']
    gamma = rslds['gamma']  # (T, 4) state posteriors
    segments = meta['segments']

    if gamma.shape[0] != len(t_common):
        print(f"  WARNING: shape mismatch in {sess_dir}, skip")
        continue

    n_sessions += 1

    for cond in CONDITIONS:
        # Condition mask
        cmask = np.zeros(len(t_common), dtype=bool)
        for seg in segments:
            name = seg[0]
            if name == cond or (cond == 'PE_1' and name == 'PE'):
                cmask |= (t_common >= seg[1]) & (t_common <= seg[2])
        if cmask.sum() < 20:
            continue

        for sname, sidx in zip(state_names, state_idxs):
            w = gamma[cmask, sidx]  # (N_cond,)
            if w.sum() < 5:  # need minimal weight
                continue
            for k in V11_MODALITY_KEYS:
                raw = npz[f'z_raw_{k}'][cmask]
                weighted_mean = np.sum(w * raw) / np.sum(w)
                acc[sname][cond][k].append(float(weighted_mean))

        # Unconditional mean for this condition
        for k in V11_MODALITY_KEYS:
            raw = npz[f'z_raw_{k}'][cmask]
            uncond[cond][k].append(float(raw.mean()))

print(f"\nSessions used: {n_sessions}")

# ── Report: OTHER state emissions during baselines ────────────────────
print(f"\n{'='*90}")
print(f"  OTHER STATE EMISSION PROFILE — BASELINE CONDITIONS")
print(f"  (channel z-score, weighted by P(state=OTHER | observation))")
print(f"{'='*90}")

header = f"  {'Channel':>24s} | {'base_EO':>12s} | {'base_EC':>12s} | {'uncond EO':>12s} | {'uncond EC':>12s}"
print(header)
print(f"  {'-' * 80}")

# Sort channels by absolute magnitude of base_EO OTHER weighted mean
rows = []
for k in V11_MODALITY_KEYS:
    eo_other = np.mean(acc['OTHER']['base_EO'][k]) if acc['OTHER']['base_EO'][k] else 0.0
    ec_other = np.mean(acc['OTHER']['base_EC'][k]) if acc['OTHER']['base_EC'][k] else 0.0
    eo_all = np.mean(uncond['base_EO'][k]) if uncond['base_EO'][k] else 0.0
    ec_all = np.mean(uncond['base_EC'][k]) if uncond['base_EC'][k] else 0.0
    rows.append((k, eo_other, ec_other, eo_all, ec_all))

# Sort by max abs magnitude in OTHER baselines
rows.sort(key=lambda r: -max(abs(r[1]), abs(r[2])))
for k, eo_other, ec_other, eo_all, ec_all in rows:
    print(f"  {k:>24s} | {eo_other:+12.3f} | {ec_other:+12.3f} | "
          f"{eo_all:+12.3f} | {ec_all:+12.3f}")

# ── Compare OTHER vs COUP vs SHARED during base_EO ───────────────────
print(f"\n{'='*90}")
print(f"  base_EO: OTHER vs COUP vs SHARED emission profiles")
print(f"  (which channels distinguish the three states during resting eyes-open?)")
print(f"{'='*90}")

print(f"  {'Channel':>24s} | {'OTHER':>10s} | {'COUP':>10s} | {'SHARED':>10s} | {'dominant':>10s}")
print(f"  {'-' * 75}")

rows = []
for k in V11_MODALITY_KEYS:
    o = np.mean(acc['OTHER']['base_EO'][k]) if acc['OTHER']['base_EO'][k] else 0.0
    c = np.mean(acc['COUP']['base_EO'][k]) if acc['COUP']['base_EO'][k] else 0.0
    s = np.mean(acc['SHARED']['base_EO'][k]) if acc['SHARED']['base_EO'][k] else 0.0
    rows.append((k, o, c, s))

# Sort by which state a channel is most elevated in
rows.sort(key=lambda r: -max(r[1], r[2], r[3]))
for k, o, c, s in rows:
    vals = {'OTHER': o, 'COUP': c, 'SHARED': s}
    dom = max(vals, key=vals.get)
    print(f"  {k:>24s} | {o:+10.3f} | {c:+10.3f} | {s:+10.3f} | {dom:>10s}")

# ── Top 5 most distinctive channels for OTHER vs COUP in base_EO ──────
print(f"\n{'='*90}")
print(f"  Top channels distinguishing OTHER from COUP in base_EO")
print(f"  (OTHER minus COUP, sorted by absolute difference)")
print(f"{'='*90}")

diffs = []
for k in V11_MODALITY_KEYS:
    o = np.mean(acc['OTHER']['base_EO'][k]) if acc['OTHER']['base_EO'][k] else 0.0
    c = np.mean(acc['COUP']['base_EO'][k]) if acc['COUP']['base_EO'][k] else 0.0
    diffs.append((k, o - c, o, c))

diffs.sort(key=lambda r: -abs(r[1]))
print(f"  {'Channel':>24s} | {'OTHER-COUP':>12s} | {'OTHER':>10s} | {'COUP':>10s}")
print(f"  {'-' * 65}")
for k, diff, o, c in diffs[:15]:
    print(f"  {k:>24s} | {diff:+12.3f} | {o:+10.3f} | {c:+10.3f}")
