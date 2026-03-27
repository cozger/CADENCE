"""Semisynthetic V2: Realistic expression-band injection + visualization.

Key improvements over V1:
1. Bandpass source to 0.5-2 Hz before injecting (expression band only)
2. Inject only into smile AUs (44, 45) — highest natural coherence
3. Use 4s lag (empirical peak from y_06 cross-correlation)
4. Measure coherence on smile AUs only

Also generates comparison scalograms:
A. Ground truth shared smile (real conv_2 data)
B. Null resting state (pseudo-dyad, no injection)
C. Null with injection at various kappas
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import permutations
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, wavelet_coherence, coherence_band_summary,
    surrogate_coherence_z, AFFECT_AUS, BAND_EXPRESSION,
)
from cadence.synthetic import generate_coupling_gate

FS = 30.0
LAG_S = 4.0  # empirical peak from y_06 smile cross-correlation
KAPPAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
INJECT_AUS = [44, 45]  # smile only — 2x higher natural coherence than frown
MEASURE_AUS = [44, 45]  # match injection
COUPLING_PROFILE = {'duty_cycle': 0.25, 'event_range_s': (5.0, 15.0), 'ramp_s': 2.0}

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'wavelet_semisynthetic_v2')
os.makedirs(out_dir, exist_ok=True)


def bandpass_expression(signal_1d, fs=FS, lo=0.5, hi=2.0):
    """Bandpass to expression band before injection."""
    sos = butter(4, [lo / (fs/2), hi / (fs/2)], btype='band', output='sos')
    return sosfiltfilt(sos, signal_1d.astype(np.float64)).astype(np.float32)


def inject_expression_coupling(p1, p2, kappa, gate, lag_s, fs):
    """Inject expression-band smile coupling from P1 into P2.

    Only the 0.5-2 Hz component of P1's smile AUs is injected,
    matching the real frequency signature of smile mimicry.
    """
    lag_samples = int(lag_s * fs)
    p2_out = p2.copy()
    alpha = (kappa * gate).astype(np.float64)

    for au in INJECT_AUS:
        # Bandpass P1 to expression band before lagging
        p1_expr = bandpass_expression(p1[:, au], fs)
        p1_lagged = np.roll(p1_expr, lag_samples)
        p1_lagged[:lag_samples] = 0.0  # zero the roll-in edge

        noise_scale = np.sqrt(np.maximum(1.0 - alpha ** 2, 0.0))
        p2_out[:, au] = (alpha * p1_lagged +
                         noise_scale * p2[:, au].astype(np.float64)).astype(np.float32)

    return p2_out


def run_one_pair(p1, p2_orig, kappa, pair_idx, pair_name):
    """Single run with z-scored coherence on smile AUs only."""
    T = p1.shape[0]
    gate = generate_coupling_gate(T, FS, COUPLING_PROFILE,
                                  seed=pair_idx * 1000 + int(kappa * 100))

    p2_coupled = inject_expression_coupling(p1, p2_orig, kappa, gate, LAG_S, FS)

    scal_p1 = compute_au_cwt(p1, device='auto')
    scal_p2 = compute_au_cwt(p2_coupled, device='auto')

    # Z-scored coherence on SMILE AUs only
    z_result = surrogate_coherence_z(scal_p1, scal_p2, n_surrogates=200,
                                      aus=MEASURE_AUS,
                                      seed=pair_idx * 100 + int(kappa * 10),
                                      device='auto')

    # Also raw coherence for comparison
    coh = wavelet_coherence(scal_p1, scal_p2, au_groups={'smile': MEASURE_AUS},
                            device='auto')
    summary = coherence_band_summary(coh, scal_p1.freqs)

    return {
        'pair': pair_name, 'kappa': kappa, 'pair_idx': pair_idx,
        'expr_z': z_result['band_z']['expression'],
        'expr_coh': summary['smile']['expression']['mean'],
        'gate_corr': float(np.corrcoef(
            z_result['z'][(scal_p1.freqs >= 0.5) & (scal_p1.freqs < 2.0)].mean(axis=0),
            gate)[0, 1]) if T == len(gate) else 0.0,
    }


# ── Load sessions ────────────────────────────────────────────────────

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
xdf_files = sorted(glob.glob(os.path.join(raw_dir, '*.xdf')))

print(f"Loading {len(xdf_files)} sessions...", flush=True)
session_segments = {}
for xdf_path in xdf_files:
    name = os.path.splitext(os.path.basename(xdf_path))[0]
    try:
        sess = load_xdf_session(xdf_path)
        m = sess['markers']
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
print(f"\n{len(session_names)} sessions loaded")

# All ordered permutations
all_pairs = [(a, b) for a, b in permutations(session_names, 2)]
pairs = all_pairs
print(f"{len(pairs)} pseudo-dyad pairings")

# ── Run sweep ────────────────────────────────────────────────────────

print(f"\n{len(pairs) * len(KAPPAS)} total runs...", flush=True)

t_start = time.time()
results = []
for pair_idx, (sess_a, sess_b) in enumerate(pairs):
    p1_full, _, _ = session_segments[sess_a]
    _, p2_full, _ = session_segments[sess_b]
    T = min(p1_full.shape[0], p2_full.shape[0])

    for kappa in KAPPAS:
        r = run_one_pair(p1_full[:T], p2_full[:T], kappa, pair_idx,
                         f"{sess_a}_x_{sess_b}")
        results.append(r)

    if (pair_idx + 1) % 10 == 0:
        print(f"  {pair_idx+1}/{len(pairs)} pairs done...", flush=True)

elapsed = time.time() - t_start
print(f"Completed in {elapsed:.1f}s ({elapsed/len(results):.2f}s/run)")

# ── Analyze ──────────────────────────────────────────────────────────

kappa_z = {k: [] for k in KAPPAS}
kappa_coh = {k: [] for k in KAPPAS}
for r in results:
    kappa_z[r['kappa']].append(r['expr_z'])
    kappa_coh[r['kappa']].append(r['expr_coh'])

print(f"\n{'kappa':>6s}  {'n':>3s}  {'mean_z':>8s}  {'std_z':>8s}  {'mean_coh':>8s}  {'gate_corr':>9s}")
for kappa in KAPPAS:
    zs = kappa_z[kappa]
    cohs = kappa_coh[kappa]
    corrs = [r['gate_corr'] for r in results if r['kappa'] == kappa]
    print(f"{kappa:6.2f}  {len(zs):3d}  {np.mean(zs):+8.3f}  {np.std(zs):8.3f}  "
          f"{np.mean(cohs):8.4f}  {np.mean(corrs):+9.3f}")

for score_name, kappa_scores in [('z-scored', kappa_z), ('raw_coh', kappa_coh)]:
    print(f"\n  {score_name}:")
    null_s = kappa_scores[0.0]
    for kappa in KAPPAS:
        if kappa == 0.0: continue
        coupled = kappa_scores[kappa]
        labels = [0]*len(null_s) + [1]*len(coupled)
        scores = null_s + coupled
        auc = roc_auc_score(labels, scores)
        d = (np.mean(coupled) - np.mean(null_s)) / np.sqrt(
            (np.std(null_s)**2 + np.std(coupled)**2) / 2 + 1e-10)
        print(f"    k={kappa:.2f}: AUC={auc:.3f}  d={d:.2f}")
    all_l = [0]*len(null_s)*len([k for k in KAPPAS if k>0]) + \
            [1]*sum(len(kappa_scores[k]) for k in KAPPAS if k>0)
    # Simpler: rebuild
    al, asc = [], []
    for k in KAPPAS:
        if k == 0: continue
        al.extend([0]*len(null_s) + [1]*len(kappa_scores[k]))
        asc.extend(null_s + kappa_scores[k])
    print(f"    Overall: AUC={roc_auc_score(al, asc):.3f}")

# ── Visualization plots ──────────────────────────────────────────────

# 1. ROC + boxplot + gate correlation (same as before)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
null_z = kappa_z[0.0]
for kappa in KAPPAS:
    if kappa == 0.0: continue
    labels = [0]*len(null_z) + [1]*len(kappa_z[kappa])
    scores = null_z + kappa_z[kappa]
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, label=f'k={kappa:.2f} (AUC={auc:.2f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title(f'ROC (z-scored, smile AUs, expr-band injection)')
ax.legend(fontsize=8)

ax = axes[1]
bp = ax.boxplot([kappa_z[k] for k in KAPPAS], positions=range(len(KAPPAS)),
                widths=0.6, patch_artist=True)
for patch, k in zip(bp['boxes'], KAPPAS):
    patch.set_facecolor('#E91E63' if k > 0 else '#BDBDBD'); patch.set_alpha(0.6)
ax.set_xticks(range(len(KAPPAS)))
ax.set_xticklabels([f'{k:.2f}' for k in KAPPAS])
ax.set_xlabel('Kappa'); ax.set_ylabel('Expression-band z')
ax.set_title('Z-score distribution'); ax.axhline(0, color='gray', linewidth=0.5)

ax = axes[2]
for k in KAPPAS:
    corrs = [r['gate_corr'] for r in results if r['kappa'] == k]
    ax.scatter([k]*len(corrs), corrs, alpha=0.3, s=15, color='tab:blue')
means = [np.mean([r['gate_corr'] for r in results if r['kappa']==k]) for k in KAPPAS]
ax.plot(KAPPAS, means, 'o-', color='tab:red', linewidth=2, markersize=8)
ax.set_xlabel('Kappa'); ax.set_ylabel('Gate correlation')
ax.set_title('Temporal localization'); ax.axhline(0, color='gray', linewidth=0.5)

fig.suptitle(f'V2: Expression-band injection, smile AUs only, {len(pairs)} pairs')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'auc_results.png'), dpi=150)
plt.close(fig)

# 2. Comparison scalograms: real smile vs null vs injected
print("\nGenerating comparison scalograms...", flush=True)

# Use y_06 conv_2 as real data, y_11 as pseudo-dyad partner
sess_real = load_xdf_session(glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))[0])
m = sess_real['markers']
p1_real, p2_real, _ = extract_bl_segment(sess_real['landmarks'],
                                          m['conv_2_start'], m['conv_2_stop'])

# Find a pseudo partner
for partner in ['y11_022526', 'y_17', 'y01_021726']:
    if partner in session_segments:
        _, p2_pseudo_full, _ = session_segments[partner]
        break

T = min(p1_real.shape[0], p2_pseudo_full.shape[0])
p1 = p1_real[:T]
p2_null = p2_pseudo_full[:T]

gate = generate_coupling_gate(T, FS, COUPLING_PROFILE, seed=42)

# Three conditions
conditions = {
    'A: Real dyad (conv_2)': (p1, p2_real[:T]),
    'B: Pseudo-dyad null (k=0)': (p1, p2_null),
    'C: Pseudo + injection (k=0.2)': (p1, inject_expression_coupling(p1, p2_null, 0.2, gate, LAG_S, FS)),
    'D: Pseudo + injection (k=0.4)': (p1, inject_expression_coupling(p1, p2_null, 0.4, gate, LAG_S, FS)),
}

fig, axes = plt.subplots(len(conditions), 3, figsize=(22, 4*len(conditions)),
                         gridspec_kw={'width_ratios': [1, 1, 1.5]})

for row, (cond_name, (c_p1, c_p2)) in enumerate(conditions.items()):
    scal1 = compute_au_cwt(c_p1, device='auto')
    scal2 = compute_au_cwt(c_p2, device='auto')

    # Smile AU power for each person
    pow1 = scal1.power[:, :T, 44] + scal1.power[:, :T, 45]
    pow2 = scal2.power[:, :T, 44] + scal2.power[:, :T, 45]

    # Coherence on smile AUs
    coh = wavelet_coherence(scal1, scal2, au_groups={'smile': MEASURE_AUS}, device='auto')
    coh_map = coh['smile']['coherence'][:, :T]

    t_axis = np.arange(T) / FS

    # P1 scalogram
    ax = axes[row, 0]
    vmax = np.percentile(pow1, 95)
    ax.pcolormesh(t_axis, scal1.freqs, pow1, shading='auto', cmap='hot',
                  vmin=0, vmax=max(vmax, 1e-8))
    ax.set_yscale('log'); ax.set_ylim(0.3, 8)
    if row == 0: ax.set_title('P1 smile power', fontsize=10)
    ax.set_ylabel(f'{cond_name}\nFreq (Hz)', fontsize=8)
    for f in [0.5, 2.0]:
        ax.axhline(f, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)

    # P2 scalogram
    ax = axes[row, 1]
    ax.pcolormesh(t_axis, scal1.freqs, pow2, shading='auto', cmap='hot',
                  vmin=0, vmax=max(vmax, 1e-8))
    ax.set_yscale('log'); ax.set_ylim(0.3, 8)
    if row == 0: ax.set_title('P2 smile power', fontsize=10)
    for f in [0.5, 2.0]:
        ax.axhline(f, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)

    # Coherence spectrogram
    ax = axes[row, 2]
    ax.pcolormesh(t_axis, scal1.freqs, coh_map, shading='auto', cmap='hot',
                  vmin=0, vmax=0.9)
    ax.set_yscale('log'); ax.set_ylim(0.3, 8)
    if row == 0: ax.set_title('Smile coherence', fontsize=10)
    for f in [0.5, 2.0]:
        ax.axhline(f, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)

axes[-1, 0].set_xlabel('Time (s)')
axes[-1, 1].set_xlabel('Time (s)')
axes[-1, 2].set_xlabel('Time (s)')

fig.suptitle('Comparison: Real dyad vs Null vs Injected — smile AU scalograms + coherence',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'comparison_scalograms.png'), dpi=120)
plt.close(fig)
print("Saved: comparison_scalograms.png")

print(f"\nAll outputs in: {out_dir}")
