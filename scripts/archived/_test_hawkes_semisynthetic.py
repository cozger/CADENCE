"""Semi-synthetic validation of Hawkes coupling pipeline.

Injects event-triggered mimicry into real raw blendshapes with a known
coupling gate. Measures:
  1. Pathway detection (does the correct pathway survive FDR?)
  2. Temporal localization AUC (MMHP episodes vs ground-truth gate)
  3. Comparison with existing Stage 1 cross-product mask

Uses real y_06 blendshapes as base, injects smile mimicry (P1 smile →
P2 smile at 2.5s lag with probability kappa) during known coupling windows.
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf
import glob
import time
from sklearn.metrics import roc_auc_score

from cadence.synthetic import generate_coupling_gate
from cadence.significance.hawkes_coupling import (
    hawkes_coupling_analysis, nmf_expression_discovery,
    detect_component_events, fit_hawkes_pathway, fit_mmhp,
    MP_BLENDSHAPE_NAMES,
)

FS = 30.0

# Smile AU indices in MediaPipe (0-indexed)
SMILE_AUS = [44, 45]  # mouthSmileLeft, mouthSmileRight


def load_raw_bl(xdf_path, segment='conv_1'):
    """Load raw [0,1] blendshapes from XDF."""
    data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

    marker_times = {}
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                marker_times[v[0]] = t

    t_start = marker_times[f'{segment}_start']
    t_end = marker_times[f'{segment}_stop']

    landmarks = {}
    for stream in data:
        name = stream['info']['name'][0]
        if 'landmarks' in name.lower():
            person = 'P1' if 'P1' in name else 'P2'
            n_ch = int(stream['info']['channel_count'][0])
            if n_ch >= 52 and person not in landmarks:
                landmarks[person] = (
                    np.array(stream['time_stamps']),
                    np.array(stream['time_series'], dtype=np.float32))

    dur = t_end - t_start
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)

    result = {}
    for person in ['P1', 'P2']:
        ts, d = landmarks[person]
        m = (ts >= t_start) & (ts <= t_end)
        sig = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                        for c in range(52)], axis=1)
        result[person] = sig

    return result['P1'], result['P2'], dur


def inject_raw_event_mimicry(p1_raw, p2_raw, gate, kappa, lag_s=2.5,
                              target_aus=None, seed=42):
    """Inject event mimicry into raw [0,1] blendshapes.

    When P1 has a smile peak during a coupling window, P2 gets a
    mimicry response at t+lag with probability kappa.

    Args:
        p1_raw, p2_raw: (T, 52) raw blendshapes in [0, 1].
        gate: (T,) coupling gate in [0, 1].
        kappa: probability of mimicry per eligible P1 event.
        lag_s: response lag in seconds.
        target_aus: AU indices to inject into. Default: smile AUs.

    Returns:
        p2_coupled: (T, 52) with injected mimicry.
        n_injected: number of mimicry events injected.
        injection_times: (n,) times of injected events in seconds.
    """
    from scipy.signal import find_peaks

    if target_aus is None:
        target_aus = SMILE_AUS

    rng = np.random.default_rng(seed)
    T = min(p1_raw.shape[0], p2_raw.shape[0])
    p2_coupled = p2_raw[:T].copy()
    lag_samp = int(lag_s * FS)

    # Build P1 smile composite for event detection
    p1_smile = np.sum(p1_raw[:T, target_aus], axis=1)

    # Detect P1 smile peaks (prominence ≥ 0.15 in raw composite)
    peaks, props = find_peaks(p1_smile, prominence=0.15,
                              distance=int(2.0 * FS))

    n_injected = 0
    injection_times = []

    for pk in peaks:
        # Check coupling gate
        if gate[pk] < 0.5:
            continue

        # Probabilistic triggering
        if rng.random() > kappa:
            continue

        # Response center in P2
        resp_center = pk + lag_samp
        if resp_center >= T - int(FS):
            continue

        # Inject a Gaussian-enveloped smile response
        half_w = int(0.5 * FS)  # 0.5s half-width
        env = np.exp(-np.linspace(-2, 2, 2 * half_w + 1) ** 2 / 2)

        # Response amplitude: scale to be a visible smile (0.2-0.5 range)
        amp = rng.uniform(0.2, 0.5)

        inj_start = max(0, resp_center - half_w)
        inj_end = min(T, resp_center + half_w + 1)
        env_start = inj_start - (resp_center - half_w)
        env_end = env_start + (inj_end - inj_start)

        for au in target_aus:
            p2_coupled[inj_start:inj_end, au] += amp * env[env_start:env_end]

        # Clip to [0, 1]
        p2_coupled[inj_start:inj_end] = np.clip(
            p2_coupled[inj_start:inj_end], 0, 1)

        n_injected += 1
        injection_times.append(resp_center / FS)

    return p2_coupled, n_injected, np.array(injection_times)


def compute_tl_auc(episodes, gate, duration, fs=FS):
    """Compute temporal localization AUC.

    Converts MMHP episodes and ground-truth gate into binary masks,
    computes ROC AUC.
    """
    T = int(duration * fs)
    gate_binary = (gate[:T] > 0.5).astype(float)

    # Build predicted mask from episodes
    pred_mask = np.zeros(T)
    for ep in episodes:
        s = int(ep.start_s * fs)
        e = int(ep.end_s * fs)
        pred_mask[max(0, s):min(T, e)] = 1.0

    # Need both classes present
    if gate_binary.sum() == 0 or gate_binary.sum() == T:
        return 0.5

    return roc_auc_score(gate_binary, pred_mask)


# ── Main ──────────────────────────────────────────────────────────────

xdf_path = glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0]
print(f"Loading: {xdf_path}")
p1_base, p2_base, dur = load_raw_bl(xdf_path, 'conv_1')
T = min(p1_base.shape[0], p2_base.shape[0])
print(f"Base data: {T} samples, {dur:.1f}s\n")

# Coupling gate: 30% duty cycle, 10-30s episodes
gate_profile = {'duty_cycle': 0.30, 'event_range_s': (10, 30), 'ramp_s': 2.0}
gate = generate_coupling_gate(T, FS, gate_profile, seed=999)
gate_frac = (gate > 0.5).mean()
print(f"Coupling gate: {gate_frac:.1%} duty cycle")

# Sweep kappa
print(f"\n{'kappa':>6} {'injected':>8} {'n_pw':>5} {'best_α':>8} "
      f"{'best_pathway':>15} {'TL_AUC':>7} {'elapsed':>8}")
print("-" * 70)

for kappa in [0.0, 0.3, 0.5, 0.7, 1.0]:
    results_per_seed = []

    for seed in range(5):
        p2_coupled, n_inj, inj_times = inject_raw_event_mimicry(
            p1_base, p2_base, gate, kappa, lag_s=2.5, seed=seed * 100)

        t0 = time.time()
        result = hawkes_coupling_analysis(
            p1_base[:T], p2_coupled[:T], FS,
            n_components=6, seed=42)
        elapsed = time.time() - t0

        # Find best pathway (highest alpha = strongest triggering)
        if result.pathways:
            best = max(result.pathways, key=lambda x: x.alpha)
            best_p = best.alpha  # repurpose column for alpha
            best_label = f"{best.source_component}→{best.target_component}"

            # TL AUC from best pathway
            tl_auc = 0.5
            if best.episodes:
                tl_auc = compute_tl_auc(best.episodes, gate, dur)
        else:
            best_p = 0.0
            best_label = "none"
            tl_auc = 0.5

        results_per_seed.append({
            'n_inj': n_inj,
            'n_sig': result.n_significant,
            'best_p': best_p,
            'best_label': best_label,
            'tl_auc': tl_auc,
            'elapsed': elapsed,
        })

    # Average over seeds
    avg_inj = np.mean([r['n_inj'] for r in results_per_seed])
    avg_sig = np.mean([r['n_sig'] for r in results_per_seed])
    avg_p = np.mean([r['best_p'] for r in results_per_seed])
    avg_auc = np.mean([r['tl_auc'] for r in results_per_seed])
    avg_elapsed = np.mean([r['elapsed'] for r in results_per_seed])
    labels = [r['best_label'] for r in results_per_seed]
    most_common = max(set(labels), key=labels.count)

    print(f"{kappa:6.1f} {avg_inj:8.1f} {avg_sig:5.1f} {avg_p:8.4f} "
          f"{most_common:>15} {avg_auc:7.3f} {avg_elapsed:7.1f}s")

    for i, r in enumerate(results_per_seed):
        print(f"        seed={i}: inj={r['n_inj']}, n_pw={r['n_sig']}, "
              f"alpha={r['best_p']:.4f}, pathway={r['best_label']}, "
              f"AUC={r['tl_auc']:.3f}")

print(f"\nDone.")
