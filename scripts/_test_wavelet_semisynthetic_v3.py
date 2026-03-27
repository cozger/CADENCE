"""Semisynthetic V3: Inject REAL smile waveforms into pseudo-dyad raw signals.

Takes the actual 52-AU smile delta waveform extracted from y_06 ground truth
smiles and stamps it into both participants' raw blendshapes at controlled
times and lags. No subsetting, no bandpassing — the injection IS a real smile.
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import permutations

from scripts.run_session_v6 import load_xdf_session, extract_bl_segment
from cadence.significance.bl_wavelet import (
    compute_au_cwt, wavelet_coherence, coherence_band_summary,
    surrogate_coherence_z, AFFECT_AUS,
)

FS = 30.0
LAG_S = 4.0
KAPPAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'v6', 'wavelet_semisynthetic_v3')
os.makedirs(out_dir, exist_ok=True)

# Load the real smile delta waveform (52 AUs, 6s @ 30Hz)
smile_delta = np.load(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'v6', 'y_06', 'coupling_patterns', 'smile_delta_avg.npy'))
# Onset is at sample 60 (2s into the 6s window)
SMILE_LEN = smile_delta.shape[0]  # 180 samples = 6s
SMILE_ONSET = int(2 * FS)  # onset at 2s into window
print(f"Smile waveform: {smile_delta.shape}, onset at sample {SMILE_ONSET}")


def inject_smile_events(p1, p2, kappa, event_times_p1, lag_samples, rng):
    """Stamp real smile waveforms into both participants.

    For each event time in P1:
      - Add kappa * smile_delta to P1 at event_time
      - Add kappa * smile_delta to P2 at event_time + lag (mimicry)
    Clip to [0, 1] to keep blendshape range valid.
    """
    p1_out = p1.copy().astype(np.float64)
    p2_out = p2.copy().astype(np.float64)
    T = p1.shape[0]

    for t_p1 in event_times_p1:
        # P1: stamp smile at t_p1
        s1 = t_p1 - SMILE_ONSET
        e1 = s1 + SMILE_LEN
        if s1 >= 0 and e1 < T:
            p1_out[s1:e1] += kappa * smile_delta

        # P2: stamp smile at t_p1 + lag (mimicry response)
        t_p2 = t_p1 + lag_samples
        s2 = t_p2 - SMILE_ONSET
        e2 = s2 + SMILE_LEN
        if s2 >= 0 and e2 < T:
            # Slight amplitude jitter for realism
            jitter = 0.8 + 0.4 * rng.random()
            p2_out[s2:e2] += kappa * jitter * smile_delta

    np.clip(p1_out, 0, 1, out=p1_out)
    np.clip(p2_out, 0, 1, out=p2_out)
    return p1_out.astype(np.float32), p2_out.astype(np.float32)


def generate_smile_event_times(T, fs, rate_per_min=3.0, seed=42):
    """Generate random smile onset times at a given rate."""
    rng = np.random.default_rng(seed)
    dur_s = T / fs
    n_events = max(1, int(rate_per_min * dur_s / 60))
    # Random times with minimum 8s spacing
    times = []
    for _ in range(n_events * 3):
        t = rng.integers(int(5*fs), T - int(8*fs))
        if all(abs(t - existing) > int(8*fs) for existing in times):
            times.append(t)
        if len(times) >= n_events:
            break
    return sorted(times)


def run_one_pair(p1, p2, kappa, pair_idx, pair_name):
    """Single run: inject smile events, compute z-scored coherence."""
    T = p1.shape[0]
    rng = np.random.default_rng(pair_idx * 1000 + int(kappa * 100))
    lag_samples = int(LAG_S * FS)

    event_times = generate_smile_event_times(T, FS, rate_per_min=3.0,
                                              seed=pair_idx * 10 + int(kappa * 100))

    if kappa > 0:
        p1_inj, p2_inj = inject_smile_events(p1, p2, kappa, event_times, lag_samples, rng)
    else:
        p1_inj, p2_inj = p1, p2

    scal_p1 = compute_au_cwt(p1_inj, device='auto')
    scal_p2 = compute_au_cwt(p2_inj, device='auto')

    # Z-scored coherence on ALL affect AUs
    z_result = surrogate_coherence_z(scal_p1, scal_p2, n_surrogates=200,
                                      aus=AFFECT_AUS,
                                      seed=pair_idx * 50 + int(kappa * 10),
                                      device='auto')

    # Raw coherence too
    coh = wavelet_coherence(scal_p1, scal_p2, device='auto')
    summary = coherence_band_summary(coh, scal_p1.freqs)

    return {
        'pair': pair_name, 'kappa': kappa, 'pair_idx': pair_idx,
        'expr_z': z_result['band_z']['expression'],
        'expr_coh': summary['affect']['expression']['mean'],
        'n_events': len(event_times),
    }


# ── Load sessions ────────────────────────────────────────────────────

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'raw sessions')
print("Loading sessions...", flush=True)
session_segments = {}
for xdf_path in sorted(glob.glob(os.path.join(raw_dir, '*.xdf'))):
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
    except Exception:
        pass

session_names = sorted(session_segments.keys())
pairs = [(a, b) for a, b in permutations(session_names, 2)]
print(f"{len(session_names)} sessions, {len(pairs)} pairs")

# ── Run ──────────────────────────────────────────────────────────────

print(f"\n{len(pairs) * len(KAPPAS)} runs...", flush=True)
t_start = time.time()

results = []
for pair_idx, (sa, sb) in enumerate(pairs):
    p1_full, _, _ = session_segments[sa]
    _, p2_full, _ = session_segments[sb]
    T = min(p1_full.shape[0], p2_full.shape[0])
    for kappa in KAPPAS:
        r = run_one_pair(p1_full[:T], p2_full[:T], kappa, pair_idx,
                         f"{sa}_x_{sb}")
        results.append(r)
    if (pair_idx + 1) % 10 == 0:
        print(f"  {pair_idx+1}/{len(pairs)}...", flush=True)

elapsed = time.time() - t_start
print(f"Done in {elapsed:.1f}s ({elapsed/len(results):.2f}s/run)")

# ── Analyze ──────────────────────────────────────────────────────────

kappa_z = {k: [] for k in KAPPAS}
kappa_coh = {k: [] for k in KAPPAS}
for r in results:
    kappa_z[r['kappa']].append(r['expr_z'])
    kappa_coh[r['kappa']].append(r['expr_coh'])

print(f"\n{'kappa':>6s}  {'n':>3s}  {'mean_z':>8s}  {'std_z':>8s}  {'mean_coh':>8s}")
for k in KAPPAS:
    print(f"{k:6.2f}  {len(kappa_z[k]):3d}  {np.mean(kappa_z[k]):+8.3f}  "
          f"{np.std(kappa_z[k]):8.3f}  {np.mean(kappa_coh[k]):8.4f}")

for sname, ks in [('z-scored', kappa_z), ('raw_coh', kappa_coh)]:
    print(f"\n  {sname}:")
    null = ks[0.0]
    for k in KAPPAS:
        if k == 0: continue
        coupled = ks[k]
        labels = [0]*len(null) + [1]*len(coupled)
        scores = null + coupled
        auc = roc_auc_score(labels, scores)
        d = (np.mean(coupled)-np.mean(null)) / np.sqrt(
            (np.std(null)**2+np.std(coupled)**2)/2 + 1e-10)
        print(f"    k={k:.2f}: AUC={auc:.3f}  d={d:+.2f}")
    al = []; asc = []
    for k in KAPPAS:
        if k == 0: continue
        al.extend([0]*len(null)+[1]*len(ks[k]))
        asc.extend(null+ks[k])
    print(f"    Overall: AUC={roc_auc_score(al,asc):.3f}")

# ── Plots ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
null_z = kappa_z[0.0]
for k in KAPPAS:
    if k == 0: continue
    labels = [0]*len(null_z)+[1]*len(kappa_z[k])
    scores = null_z+kappa_z[k]
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, label=f'k={k:.2f} (AUC={auc:.2f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC (z-scored, real smile injection)')
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
    cohs = kappa_coh[k]
    ax.scatter([k]*len(cohs), cohs, alpha=0.3, s=15, color='tab:blue')
means = [np.mean(kappa_coh[k]) for k in KAPPAS]
ax.plot(KAPPAS, means, 'o-', color='tab:red', linewidth=2, markersize=8)
ax.set_xlabel('Kappa'); ax.set_ylabel('Expression-band coh')
ax.set_title('Raw coherence vs kappa')

fig.suptitle(f'V3: Real smile waveform injection, all affect AUs, {len(pairs)} pairs')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'auc_results.png'), dpi=150)
plt.close(fig)

# ── Comparison scalograms ────────────────────────────────────────────

print("\nGenerating comparison scalograms...", flush=True)

sess_real = load_xdf_session(glob.glob(os.path.join(raw_dir, '*y_06*.xdf'))[0])
mr = sess_real['markers']
p1_real, p2_real, _ = extract_bl_segment(sess_real['landmarks'],
                                          mr['conv_2_start'], mr['conv_2_stop'])

for partner in ['y11_022526', 'y_17']:
    if partner in session_segments:
        _, p2_pseudo_full, _ = session_segments[partner]
        break

T = min(p1_real.shape[0], p2_pseudo_full.shape[0])
p1 = p1_real[:T]
p2_null = p2_pseudo_full[:T]

rng = np.random.default_rng(42)
events = generate_smile_event_times(T, FS, rate_per_min=3.0, seed=42)
lag_samp = int(LAG_S * FS)

conditions = {
    'Real dyad (conv_2)': (p1, p2_real[:T]),
    'Pseudo null (k=0)': (p1, p2_null),
}
for k in [0.2, 0.4]:
    p1i, p2i = inject_smile_events(p1, p2_null, k, events, lag_samp, rng)
    conditions[f'Injected k={k}'] = (p1i, p2i)

fig, axes = plt.subplots(len(conditions), 3, figsize=(22, 4*len(conditions)),
                         gridspec_kw={'width_ratios': [1, 1, 1.5]})

for row, (cname, (cp1, cp2)) in enumerate(conditions.items()):
    s1 = compute_au_cwt(cp1, device='auto')
    s2 = compute_au_cwt(cp2, device='auto')

    # Affect AU power summed
    pow1 = sum(s1.power[:, :T, au] for au in AFFECT_AUS)
    pow2 = sum(s2.power[:, :T, au] for au in AFFECT_AUS)
    coh = wavelet_coherence(s1, s2, device='auto')
    coh_map = coh['affect']['coherence'][:, :T]

    t_ax = np.arange(T) / FS
    vmax_p = max(np.percentile(pow1, 95), np.percentile(pow2, 95))

    ax = axes[row, 0]
    ax.pcolormesh(t_ax, s1.freqs, pow1, shading='auto', cmap='hot', vmin=0, vmax=vmax_p)
    ax.set_yscale('log'); ax.set_ylim(0.3, 8)
    ax.set_ylabel(f'{cname}\nFreq (Hz)', fontsize=8)
    if row == 0: ax.set_title('P1 affect power')
    for f in [0.5, 2.0]: ax.axhline(f, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)

    ax = axes[row, 1]
    ax.pcolormesh(t_ax, s1.freqs, pow2, shading='auto', cmap='hot', vmin=0, vmax=vmax_p)
    ax.set_yscale('log'); ax.set_ylim(0.3, 8)
    if row == 0: ax.set_title('P2 affect power')
    for f in [0.5, 2.0]: ax.axhline(f, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)

    ax = axes[row, 2]
    ax.pcolormesh(t_ax, s1.freqs, coh_map, shading='auto', cmap='hot', vmin=0, vmax=0.8)
    ax.set_yscale('log'); ax.set_ylim(0.3, 8)
    if row == 0: ax.set_title('Affect coherence')
    for f in [0.5, 2.0]: ax.axhline(f, color='cyan', linewidth=0.5, linestyle='--', alpha=0.5)

for ax in axes[-1]: ax.set_xlabel('Time (s)')
fig.suptitle('V3: Real smile injection — affect AU scalograms + coherence')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'comparison_scalograms.png'), dpi=120)
plt.close(fig)
print("Saved comparison_scalograms.png")

print(f"\nAll outputs in: {out_dir}")
