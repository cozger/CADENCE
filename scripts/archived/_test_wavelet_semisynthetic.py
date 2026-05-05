"""Semisynthetic AUC validation of wavelet coherence — multi-pair design.

Uses multiple pseudo-dyad pairings for proper null variance.
Each pairing contributes one null (kappa=0) and multiple coupled runs.
AUC computed across all pairings.
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import Parallel, delayed
from itertools import combinations

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, wavelet_coherence, coherence_band_summary,
    surrogate_coherence_z, AFFECT_AUS
)
from cadence.synthetic import generate_coupling_gate

FS = 30.0
LAG_S = 2.0
KAPPAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
INJECT_AUS = [44, 45, 30, 31]
COUPLING_PROFILE = {'duty_cycle': 0.30, 'event_range_s': (5.0, 20.0), 'ramp_s': 2.0}

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'wavelet_semisynthetic')
os.makedirs(out_dir, exist_ok=True)


def inject_au_coupling(p1, p2, kappa, gate, lag_s, fs, aus):
    """Inject P1->P2 coupling into specific AUs with variance preservation."""
    lag_samples = int(lag_s * fs)
    p2_out = p2.copy()
    alpha = (kappa * gate).astype(np.float64)
    for au in aus:
        p1_lagged = np.roll(p1[:, au].astype(np.float64), lag_samples)
        p1_lagged[:lag_samples] = p2[:lag_samples, au]
        noise_scale = np.sqrt(np.maximum(1.0 - alpha ** 2, 0.0))
        p2_out[:, au] = (alpha * p1_lagged + noise_scale * p2[:, au].astype(np.float64)).astype(np.float32)
    return p2_out


def run_one_pair(p1, p2_orig, kappa, seed, pair_name):
    """Single run: inject coupling, compute z-scored coherence + gate correlation."""
    T = p1.shape[0]
    gate = generate_coupling_gate(T, FS, COUPLING_PROFILE,
                                  seed=seed * 1000 + int(kappa * 100))

    p2_coupled = inject_au_coupling(p1, p2_orig, kappa, gate, LAG_S, FS, INJECT_AUS)

    scal_p1 = compute_au_cwt(p1, device='auto')
    scal_p2 = compute_au_cwt(p2_coupled, device='auto')

    # Z-scored coherence against 200 circular-shift surrogates
    z_result = surrogate_coherence_z(scal_p1, scal_p2, n_surrogates=200,
                                      seed=seed * 100 + int(kappa * 10),
                                      device='auto')

    expr_z = z_result['band_z']['expression']

    # Also get raw coherence for comparison
    coh = wavelet_coherence(scal_p1, scal_p2, device='auto')
    summary = coherence_band_summary(coh, scal_p1.freqs)
    expr_coh = summary['affect']['expression']['mean']

    # Gate correlation from z timecourse
    z_tc = z_result['z']  # (n_freqs, T)
    freqs = scal_p1.freqs
    expr_mask = (freqs >= 0.5) & (freqs < 2.0)
    z_expr_tc = z_tc[expr_mask].mean(axis=0) if expr_mask.sum() > 0 else np.zeros(T)
    gate_corr = float(np.corrcoef(z_expr_tc, gate)[0, 1]) if len(z_expr_tc) == len(gate) else 0.0

    return {
        'pair': pair_name, 'kappa': kappa, 'seed': seed,
        'expr_z': expr_z, 'expr_coh': expr_coh, 'gate_corr': gate_corr,
    }


# ── Load all sessions and find conv segments ─────────────────────────

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = sorted(glob.glob(os.path.join(raw_dir, '*.xdf')))

print(f"Loading {len(xdf_files)} sessions...", flush=True)

session_segments = {}  # session_name -> (p1_bl, p2_bl)
for xdf_path in xdf_files:
    name = os.path.splitext(os.path.basename(xdf_path))[0]
    try:
        sess = load_xdf_session(xdf_path)
        m = sess['markers']
        # Try conv_2 first, then conv_1
        for seg in ['conv_2', 'conv_1']:
            t0 = m.get(f'{seg}_start')
            t1 = m.get(f'{seg}_stop')
            if t0 is not None and (t1 - t0) >= 120:
                p1, p2, dur = extract_bl_segment(sess['landmarks'], t0, t1)
                if p1 is not None:
                    session_segments[name] = (p1, p2, dur)
                    print(f"  {name}: {seg} {dur:.0f}s")
                    break
    except Exception as e:
        print(f"  {name}: SKIP ({e})")

session_names = sorted(session_segments.keys())
print(f"\n{len(session_names)} sessions loaded: {session_names}")

# ── Build pseudo-dyad pairings (P1 from session A, P2 from session B) ─

# All ordered cross-session pairings (P1 from A x P2 from B != P1 from B x P2 from A)
from itertools import permutations
all_pairs = [(a, b) for a, b in permutations(session_names, 2)]
pairs = all_pairs  # use all
print(f"\n{len(pairs)} pseudo-dyad pairings:")
for a, b in pairs:
    print(f"  P1({a}) x P2({b})")

# ── Run all pairings x kappas ────────────────────────────────────────

jobs = []
for pair_idx, (sess_a, sess_b) in enumerate(pairs):
    p1_full, _, _ = session_segments[sess_a]
    _, p2_full, _ = session_segments[sess_b]
    T = min(p1_full.shape[0], p2_full.shape[0])
    p1 = p1_full[:T]
    p2 = p2_full[:T]
    pair_name = f"{sess_a}_x_{sess_b}"

    for kappa in KAPPAS:
        jobs.append((p1, p2, kappa, pair_idx, pair_name))

print(f"\n{len(jobs)} total runs ({len(pairs)} pairs x {len(KAPPAS)} kappas)...", flush=True)

t_start = time.time()
results = Parallel(n_jobs=8, prefer='threads')(
    delayed(run_one_pair)(p1, p2, kappa, seed, pair_name)
    for p1, p2, kappa, seed, pair_name in jobs
)
elapsed = time.time() - t_start
print(f"Completed in {elapsed:.1f}s ({elapsed/len(jobs):.2f}s/run)")

# ── Analyze ──────────────────────────────────────────────────────────

print(f"\n{'kappa':>6s}  {'n':>3s}  {'mean_z':>8s}  {'std_z':>8s}  {'mean_coh':>8s}  {'gate_corr':>9s}")
print(f"{'-----':>6s}  {'---':>3s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'---------':>9s}")

kappa_zscores = {k: [] for k in KAPPAS}
kappa_cohscores = {k: [] for k in KAPPAS}
for r in results:
    kappa_zscores[r['kappa']].append(r['expr_z'])
    kappa_cohscores[r['kappa']].append(r['expr_coh'])

for kappa in KAPPAS:
    zs = kappa_zscores[kappa]
    cohs = kappa_cohscores[kappa]
    corrs = [r['gate_corr'] for r in results if r['kappa'] == kappa]
    print(f"{kappa:6.2f}  {len(zs):3d}  {np.mean(zs):+8.3f}  {np.std(zs):8.3f}  "
          f"{np.mean(cohs):8.4f}  {np.mean(corrs):+9.3f}")

# Per-kappa AUC using BOTH z-score and raw coherence
for score_name, kappa_scores in [('z-scored', kappa_zscores), ('raw_coh', kappa_cohscores)]:
    print(f"\n  {score_name} detection:")
    print(f"  {'kappa':>6s}  {'AUC':>6s}  {'sep_d':>6s}")
    print(f"  {'-----':>6s}  {'------':>6s}  {'------':>6s}")

    null_scores = kappa_scores[0.0]
    all_labels = []
    all_scores_list = []

    for kappa in KAPPAS:
        if kappa == 0.0:
            continue
        coupled = kappa_scores[kappa]
        labels = [0] * len(null_scores) + [1] * len(coupled)
        scores = null_scores + coupled
        auc = roc_auc_score(labels, scores)
        d = (np.mean(coupled) - np.mean(null_scores)) / np.sqrt(
            (np.std(null_scores)**2 + np.std(coupled)**2) / 2 + 1e-10)
        print(f"  {kappa:6.2f}  {auc:6.3f}  {d:6.2f}")
        all_labels.extend(labels)
        all_scores_list.extend(scores)

    if len(set(all_labels)) >= 2:
        overall_auc = roc_auc_score(all_labels, all_scores_list)
        print(f"  Overall AUC: {overall_auc:.3f}")

# ── Plots ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC curves (z-scored)
ax = axes[0]
null_z = kappa_zscores[0.0]
for kappa in KAPPAS:
    if kappa == 0.0:
        continue
    labels = [0] * len(null_z) + [1] * len(kappa_zscores[kappa])
    scores = null_z + kappa_zscores[kappa]
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, label=f'k={kappa:.2f} (AUC={auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC (z-scored): {len(pairs)} pseudo-dyad pairings')
ax.legend(fontsize=8)

# Z-score distributions
ax = axes[1]
positions = list(range(len(KAPPAS)))
bp_data = [kappa_zscores[k] for k in KAPPAS]
bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True)
for patch, kappa in zip(bp['boxes'], KAPPAS):
    patch.set_facecolor('#E91E63' if kappa > 0 else '#BDBDBD')
    patch.set_alpha(0.6)
ax.set_xticks(positions)
ax.set_xticklabels([f'{k:.2f}' for k in KAPPAS])
ax.set_xlabel('Kappa')
ax.set_ylabel('Expression-band z-score')
ax.set_title('Z-scored coherence per kappa')
ax.axhline(0, color='gray', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

# Gate correlation
ax = axes[2]
for kappa in KAPPAS:
    corrs = [r['gate_corr'] for r in results if r['kappa'] == kappa]
    ax.scatter([kappa] * len(corrs), corrs, alpha=0.4, s=20, color='tab:blue')
means = [np.mean([r['gate_corr'] for r in results if r['kappa'] == k]) for k in KAPPAS]
ax.plot(KAPPAS, means, 'o-', color='tab:red', linewidth=2, markersize=8)
ax.set_xlabel('Kappa')
ax.set_ylabel('Correlation with coupling gate')
ax.set_title('Temporal localization quality')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5)

fig.suptitle(f'Wavelet Coherence Semisynthetic: {len(pairs)} pseudo-dyad pairings, '
             f'{len(KAPPAS)} kappas', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'wavelet_semisynthetic_multipair.png'), dpi=150)
plt.close(fig)
print(f"\nSaved: {os.path.join(out_dir, 'wavelet_semisynthetic_multipair.png')}")
