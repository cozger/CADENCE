"""CADENCE V2 Validation Battery — Feature-Specific Sparse Coupling Detection.

Tests the V2 doubly-sparse pipeline's ability to detect coupling at specific
features (wavelet frequency bins, PCA components, body segments) with
sensitivity sweeps across all modalities.

Categories:
  A. Null control (FPR under various confounders)
  B. EEG wavelet sensitivity (kappa sweep, frequency-specific detection)
  C. Interbrain cross-modal sensitivity (phase sync kappa sweep)
  D. Behavioral modality sensitivity (BL, Pose kappa sweeps)
  E. Feature specificity (multi-frequency, multi-ROI verification)
  F. Temporal pattern robustness (sustained, sparse, single event)
  G. Directionality calibration (one-way vs bidirectional)

Usage:
    python scripts/run_validation_battery.py --quick   --output results/battery_quick
    python scripts/run_validation_battery.py           --output results/battery_default
    python scripts/run_validation_battery.py --full    --output results/battery_full
"""

import os
import sys
import json
import time
import traceback
import argparse
import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.synthetic import (
    build_synthetic_session_v2,
    build_synthetic_wavelet_session,
    build_synthetic_interbrain_session,
    generate_coupling_gate,
    _generate_single_band_gate,
)
from cadence.data.alignment import _ensure_v2_features
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import (
    MOD_SHORT_V2, SYNTH_MODALITY_CONFIG_V2, COUPLING_PROFILES_V2,
    WAVELET_CENTER_FREQS, EEG_ROIS, EEG_ROI_NAMES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# V2 behavioral modalities (for build_synthetic_session_v2 tests)
BEHAVIORAL_MODS = ['eeg_wavelet', 'ecg_features_v2', 'blendshapes_v2', 'pose_features']
MOD_LABELS = {m: MOD_SHORT_V2.get(m, m) for m in BEHAVIORAL_MODS}
MOD_LABELS['eeg_interbrain'] = 'EEGib'

# Expected pathways by coupling type.
# EEG wavelet coupling (P1 raw EEG -> P2 raw EEG at specific freq/ROI):
#   Pipeline computes wavelet + interbrain features, detects via either pathway.
EXPECTED_EEG = {
    ('eeg_interbrain', 'eeg_wavelet'),
    ('eeg_wavelet', 'eeg_wavelet'),
}

# Interbrain coupling (shared phase sync -> blendshapes):
#   Pipeline detects via interbrain source or blendshape reverse prediction.
EXPECTED_INTERBRAIN = {
    ('eeg_interbrain', 'eeg_wavelet'),
    ('eeg_interbrain', 'blendshapes_v2'),
    ('eeg_wavelet', 'eeg_wavelet'),       # shared phase creates direct wavelet coupling
    ('blendshapes_v2', 'eeg_wavelet'),
}

# Global incremental state
_INCREMENTAL_PATH = None
_COMPLETED_SESSIONS = set()
_SKIP_SUBS = set()


# ===================================================================
# Infrastructure
# ===================================================================

def _prefetch_loop(items, build_fn):
    """Yield (item, session) with one-ahead CPU prefetching."""
    items = list(items)
    if not items:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(build_fn, items[0])
        for i, item in enumerate(items):
            session = future.result()
            if i + 1 < len(items):
                future = pool.submit(build_fn, items[i + 1])
            yield item, session
            del session


def _save_incremental(result_dict):
    """Append one result to the incremental JSONL file."""
    if _INCREMENTAL_PATH is None:
        return
    entry = {k: v for k, v in result_dict.items()
             if k not in ('pathway_details', 'feature_info')}
    with open(_INCREMENTAL_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def _header(title, detail):
    print(f"\n{'=' * 60}")
    print(f"{title}  ({detail})")
    print(f"{'=' * 60}")


def _save_category(label, results, output_dir):
    """Save completed category to its own JSON file."""
    path = os.path.join(output_dir, f'category_{label}.json')
    data = [{k: v for k, v in r.items()
             if k not in ('pathway_details', 'feature_info')}
            for r in results]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    gc.collect()
    torch.cuda.empty_cache()


def analyze(name, session, config, direction='p1_to_p2',
            expected_pathways=None, expected_freq_idx=None):
    """Run one V2 scenario, classify TP/FN/FP/TN.

    Args:
        name: Unique scenario name (used for resume).
        session: Synthetic session dict (must be ready for V2 pipeline).
        config: Pipeline config dict.
        direction: 'p1_to_p2' or 'p2_to_p1'.
        expected_pathways: set of (src, tgt) tuples that SHOULD be detected.
            Any significant pathway not in this set is a false positive.
        expected_freq_idx: If set, check if selected features include this
            wavelet frequency bin (±2 bins). Stored as 'freq_recovered' in output.

    Returns:
        dict with classification (tp/fn/fp/tn), metrics, and feature info.
        None if the scenario was already completed (--resume mode).
    """
    if name in _COMPLETED_SESSIONS:
        return None

    t0 = time.time()
    estimator = CouplingEstimator(config)
    result = estimator.analyze_session(session, direction)
    elapsed = time.time() - t0

    if expected_pathways is None:
        expected_pathways = set()

    tp, fn, fp, tn = [], [], [], []
    pathway_details = {}

    for key, is_sig in result.pathway_significant.items():
        src, tgt = key
        pkey = f'{src}->{tgt}'
        expected = key in expected_pathways
        dr2 = float(np.nanmean(result.pathway_dr2.get(key, [0])))
        pathway_details[pkey] = {'significant': bool(is_sig), 'dr2': dr2}

        if expected:
            (tp if is_sig else fn).append(pkey)
        else:
            (fp if is_sig else tn).append(pkey)

    # Feature frequency recovery check
    freq_recovered = None
    if expected_freq_idx is not None and result.discovery is not None:
        n_freqs = config.get('wavelet', {}).get('n_frequencies', 20)
        n_rois = len(EEG_ROI_NAMES)
        freq_recovered = False
        for key in [('eeg_interbrain', 'eeg_wavelet'),
                    ('eeg_wavelet', 'eeg_wavelet')]:
            selected = result.discovery.selected_features.get(key, [])
            clusters = result.discovery.feature_clusters.get(key, {})
            if not selected:
                continue
            # Map cluster indices to original feature indices
            orig = set()
            for idx in selected:
                if clusters and idx in clusters:
                    orig.update(clusters[idx])
                else:
                    orig.add(idx)
            # Decode frequency bins from feature indices
            for feat in orig:
                freq_bin = (feat % (n_freqs * n_rois)) // n_rois
                if abs(freq_bin - expected_freq_idx) <= 2:
                    freq_recovered = True
                    break
            if freq_recovered:
                break

    # Feature recovery info from discovery
    feature_info = {}
    if result.discovery is not None:
        for key in expected_pathways:
            sel = result.discovery.selected_features.get(key, [])
            if sel:
                feature_info[f'{key[0]}->{key[1]}'] = {
                    'n_selected': len(sel),
                    'features': sel[:20],
                }

    # Temporal AUC for significant pathways with known coupling gates
    temporal_auc = {}
    coupling_gates = session.get('coupling_gates', {})
    if coupling_gates and result.pathway_dr2:
        try:
            from sklearn.metrics import roc_auc_score
            eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
            dur = session.get('duration', 0)
            eval_t = np.arange(0, dur, 1.0 / eval_rate)

            for key, dr2 in result.pathway_dr2.items():
                if not result.pathway_significant.get(key, False):
                    continue
                # Find matching gate (by modality name overlap)
                gate = None
                for gmod, g in coupling_gates.items():
                    if (gmod in key[0] or key[0] in gmod
                            or gmod in key[1] or key[1] in gmod):
                        gate = g
                        break
                if gate is None:
                    continue
                # Resample gate to eval rate
                gate_t = np.arange(len(gate)) * dur / max(len(gate), 1)
                gate_r = np.interp(eval_t, gate_t, gate)
                gate_bin = (gate_r > 0.1).astype(float)
                n = min(len(dr2), len(gate_bin))
                if gate_bin[:n].sum() < 1 or gate_bin[:n].sum() > n - 1:
                    continue
                auc = roc_auc_score(gate_bin[:n], np.nan_to_num(dr2[:n]))
                temporal_auc[f'{key[0]}->{key[1]}'] = round(auc, 3)
        except Exception:
            pass

    out = {
        'name': name, 'direction': direction,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'n_sig': result.n_significant_pathways,
        'elapsed_s': round(elapsed, 1),
        'pathway_details': pathway_details,
        'feature_info': feature_info,
        'freq_recovered': freq_recovered,
        'temporal_auc': temporal_auc,
    }

    del estimator, result
    gc.collect()
    torch.cuda.empty_cache()
    _save_incremental(out)
    return out


# ===================================================================
# Session building helpers
# ===================================================================

def inject_shared_trend(session, duration, strength=0.3):
    """Add identical slow drift to both participants (confound)."""
    for mod, cfg in SYNTH_MODALITY_CONFIG_V2.items():
        hz = cfg['hz']
        n = int(duration * hz)
        t = np.arange(n) / hz
        trend = (strength * (t / duration - 0.5)
                 + 0.2 * np.sin(2 * np.pi * t / 300)).astype(np.float32)
        for p in ['p1', 'p2']:
            key = f'{p}_{mod}'
            if key not in session:
                continue
            data = session[key].copy()
            m = min(len(trend), len(data))
            data[:m] += trend[:m, None]
            mu = data.mean(axis=0, keepdims=True)
            sd = data.std(axis=0, keepdims=True) + 1e-8
            session[key] = np.clip(
                (data - mu) / sd, -10, 10).astype(np.float32)


def inject_shared_noise(session, duration, strength=0.5, seed=999):
    """Add identical instantaneous noise to both participants (confound)."""
    rng = np.random.default_rng(seed)
    for mod, cfg in SYNTH_MODALITY_CONFIG_V2.items():
        hz, n_ch = cfg['hz'], cfg['n_ch']
        n = int(duration * hz)
        noise = rng.standard_normal((n, n_ch)).astype(np.float32) * strength
        for p in ['p1', 'p2']:
            key = f'{p}_{mod}'
            if key not in session:
                continue
            data = session[key].copy()
            m = min(n, len(data))
            c = min(n_ch, data.shape[1])
            data[:m, :c] += noise[:m, :c]
            mu = data.mean(axis=0, keepdims=True)
            sd = data.std(axis=0, keepdims=True) + 1e-8
            session[key] = np.clip(
                (data - mu) / sd, -10, 10).astype(np.float32)


def build_correlated_activity_session(duration, seed=42):
    """Null session with SHARED activity envelopes (confound)."""
    null_kd = {m: 0.0 for m in BEHAVIORAL_MODS}
    session = build_synthetic_session_v2(duration, null_kd, seed=seed)

    for mod_idx, (mod, cfg) in enumerate(SYNTH_MODALITY_CONFIG_V2.items()):
        hz = cfg['hz']
        n = int(duration * hz)
        base_ch = cfg.get('base_ch', cfg['n_ch'])
        profile = COUPLING_PROFILES_V2[mod]
        act_duty = min(0.5, max(profile['duty_cycle'] * 3, 0.15))
        act_profile = dict(profile)
        act_profile['duty_cycle'] = act_duty
        shared_env = generate_coupling_gate(
            n, hz, act_profile, seed=seed + mod_idx * 100 + 7777)
        amp = (1.0 + 2.0 * shared_env)[:, None]
        for p in ['p1', 'p2']:
            key = f'{p}_{mod}'
            if key not in session:
                continue
            data = session[key].copy()
            m = min(len(amp), len(data))
            data[:m, :base_ch] *= amp[:m]
            mu = data.mean(axis=0, keepdims=True)
            sd = data.std(axis=0, keepdims=True) + 1e-8
            session[key] = np.clip(
                (data - mu) / sd, -10, 10).astype(np.float32)
    return session


def build_custom_gate(duration, hz, pattern, seed=42):
    """Build custom coupling gate for temporal pattern tests."""
    from scipy.ndimage import gaussian_filter1d
    n = int(duration * hz)
    rng = np.random.default_rng(seed)

    if pattern == 'sustained':
        return np.ones(n, dtype=np.float32)

    if pattern == 'sparse':
        return _generate_single_band_gate(n, hz, 0.05, 2.0, 8.0, 0.5, rng)

    if pattern == 'single_event':
        gate = np.zeros(n, dtype=np.float32)
        start = int(duration * 0.33 * hz)
        end = min(int(duration * 0.50 * hz), n)
        gate[start:end] = 1.0
        sigma = 2.0 * hz / 2.35
        if sigma > 0.5:
            gate = gaussian_filter1d(gate, sigma)
        return gate.astype(np.float32)

    if pattern == 'late_onset':
        gate = np.zeros(n, dtype=np.float32)
        gate[n // 2:] = 1.0
        sigma = 5.0 * hz / 2.35
        if sigma > 0.5:
            gate = gaussian_filter1d(gate, sigma)
        return gate.astype(np.float32)

    if pattern == 'early_offset':
        gate = np.zeros(n, dtype=np.float32)
        gate[:n // 2] = 1.0
        sigma = 5.0 * hz / 2.35
        if sigma > 0.5:
            gate = gaussian_filter1d(gate, sigma)
        return gate.astype(np.float32)

    raise ValueError(f"Unknown pattern: {pattern}")


def build_wavelet_custom_gate(duration, freq, roi, kappa, pattern, seed,
                               config):
    """Build wavelet session with custom temporal coupling gate.

    Injects frequency-specific EEG coupling with a controlled temporal
    pattern (sustained, sparse, single event, etc.) at the raw EEG level.
    """
    sess = build_synthetic_wavelet_session(
        duration, coupling_freq=freq, coupling_roi=roi,
        kappa=0, seed=seed)

    srate = 256
    n_samples = int(duration * srate)
    t = np.arange(n_samples) / srate
    roi_channels = EEG_ROIS[roi]
    coupling_signal = np.sin(2 * np.pi * freq * t)
    lag_samples = int(1.0 * srate)

    gate = build_custom_gate(duration, srate, pattern, seed=seed + 50000)

    for ch in roi_channels:
        p1_contrib = np.roll(
            coupling_signal * sess['p1_eeg'][:, ch], lag_samples)
        p1_contrib[:lag_samples] = 0
        alpha_t = kappa * gate[:n_samples]
        sess['p2_eeg'][:, ch] = (
            (1 - alpha_t) * sess['p2_eeg'][:, ch]
            + alpha_t * p1_contrib)

    # Re-z-score
    for ch in range(14):
        std = sess['p2_eeg'][:, ch].std()
        if std > 1e-8:
            sess['p2_eeg'][:, ch] = (
                (sess['p2_eeg'][:, ch] - sess['p2_eeg'][:, ch].mean()) / std)

    sess['coupling_gates'] = {'eeg_wavelet': gate[:n_samples]}
    return _ensure_v2_features(sess, config)


# ===================================================================
# Category A: Null Control
# ===================================================================

def category_a(duration, config, out, ns, quick=False):
    """Null sessions under various confounders. Expected: 0% FPR."""
    results = []
    null_kd = {m: 0.0 for m in BEHAVIORAL_MODS}

    # A1: Pure null — wavelet EEG session (tests interbrain FP control)
    if 'a1' not in _SKIP_SUBS:
        _header('A1: Pure Null (EEG/Interbrain)', ns['a1'])
        def _build_a1(s):
            sess = build_synthetic_wavelet_session(
                duration, coupling_freq=6.5, coupling_roi='frontal',
                kappa=0, seed=5000 + s * 1000)
            return _ensure_v2_features(sess, config)
        for s, session in _prefetch_loop(range(ns['a1']), _build_a1):
            r = analyze(f'a1_null_eeg_{s}', session, config,
                        expected_pathways=set())
            if r is None:
                continue
            r['sub'] = 'a1'
            results.append(r)
            print(f"  a1_null_eeg_{s}: FP={len(r['fp'])} ({r['elapsed_s']}s)")
    else:
        print("  [SKIPPED] A1")

    # A2: Pure null — full modality V2 session (tests behavioral FP control)
    if 'a2' not in _SKIP_SUBS:
        _header('A2: Pure Null (All Modalities)', ns['a2'])
        def _build_a2(s):
            return build_synthetic_session_v2(
                duration, null_kd, seed=6000 + s * 1000)
        for s, session in _prefetch_loop(range(ns['a2']), _build_a2):
            r = analyze(f'a2_null_all_{s}', session, config,
                        expected_pathways=set())
            if r is None:
                continue
            r['sub'] = 'a2'
            results.append(r)
            print(f"  a2_null_all_{s}: FP={len(r['fp'])} ({r['elapsed_s']}s)")
    else:
        print("  [SKIPPED] A2")

    # A3: Shared trend confound
    if 'a3' not in _SKIP_SUBS:
        _header('A3: Shared Trend', ns['a3'])
        def _build_a3(s):
            sess = build_synthetic_session_v2(
                duration, null_kd, seed=7000 + s * 1000)
            inject_shared_trend(sess, duration)
            return sess
        for s, session in _prefetch_loop(range(ns['a3']), _build_a3):
            r = analyze(f'a3_trend_{s}', session, config,
                        expected_pathways=set())
            if r is None:
                continue
            r['sub'] = 'a3'
            results.append(r)
            print(f"  a3_trend_{s}: FP={len(r['fp'])} ({r['elapsed_s']}s)")
    else:
        print("  [SKIPPED] A3")

    # A4: Correlated activity envelopes
    if 'a4' not in _SKIP_SUBS:
        _header('A4: Correlated Activity', ns['a4'])
        def _build_a4(s):
            return build_correlated_activity_session(
                duration, seed=8000 + s * 1000)
        for s, session in _prefetch_loop(range(ns['a4']), _build_a4):
            r = analyze(f'a4_corract_{s}', session, config,
                        expected_pathways=set())
            if r is None:
                continue
            r['sub'] = 'a4'
            results.append(r)
            print(f"  a4_corract_{s}: FP={len(r['fp'])} ({r['elapsed_s']}s)")
    else:
        print("  [SKIPPED] A4")

    # A5: Shared instantaneous noise
    if 'a5' not in _SKIP_SUBS:
        _header('A5: Shared Noise', ns['a5'])
        def _build_a5(s):
            sess = build_synthetic_session_v2(
                duration, null_kd, seed=9000 + s * 1000)
            inject_shared_noise(sess, duration, seed=9000 + s)
            return sess
        for s, session in _prefetch_loop(range(ns['a5']), _build_a5):
            r = analyze(f'a5_noise_{s}', session, config,
                        expected_pathways=set())
            if r is None:
                continue
            r['sub'] = 'a5'
            results.append(r)
            print(f"  a5_noise_{s}: FP={len(r['fp'])} ({r['elapsed_s']}s)")
    else:
        print("  [SKIPPED] A5")

    return results


# ===================================================================
# Category B: EEG Wavelet Sensitivity Sweep
# ===================================================================

def category_b(duration, config, out, ns, quick=False):
    """Frequency-specific EEG coupling: kappa sweep at 6.5 Hz frontal.

    Uses build_synthetic_wavelet_session which injects coupling in raw EEG
    at a specific frequency and ROI. The V2 pipeline computes wavelet and
    interbrain features, then detects via doubly-sparse feature selection.
    """
    results = []
    kappas = ([0.3, 0.5, 0.7] if quick
              else [0.2, 0.3, 0.4, 0.5, 0.6, 0.8])

    _header('B: EEG Wavelet Sensitivity',
            f'{len(kappas)} kappas x {ns["b"]} seeds')

    freq_idx = int(np.argmin(np.abs(WAVELET_CENTER_FREQS - 6.5)))

    jobs = [(kappa, s) for kappa in kappas for s in range(ns['b'])]
    def _build(params):
        kappa, s = params
        seed = 10000 + int(kappa * 1000) * 100 + s
        sess = build_synthetic_wavelet_session(
            duration, coupling_freq=6.5, coupling_roi='frontal',
            kappa=kappa, seed=seed)
        return _ensure_v2_features(sess, config)

    for (kappa, s), session in _prefetch_loop(jobs, _build):
        r = analyze(f'b_eeg_k{kappa:.2f}_{s}', session, config,
                    expected_pathways=EXPECTED_EEG,
                    expected_freq_idx=freq_idx)
        if r is None:
            continue
        r['sub'] = 'b'
        r['kappa'] = kappa
        r['modality'] = 'eeg_wavelet'
        results.append(r)
        det = 'HIT' if r['tp'] else 'miss'
        frec = ' freq_ok' if r.get('freq_recovered') else ''
        auc_str = ''
        if r.get('temporal_auc'):
            auc_vals = list(r['temporal_auc'].values())
            if auc_vals:
                auc_str = f' AUC={max(auc_vals):.3f}'
        print(f"  EEGw k={kappa:.2f} s{s}: {det}{frec}{auc_str} ({r['elapsed_s']}s)")

    return results


# ===================================================================
# Category C: Interbrain Cross-Modal Sensitivity
# ===================================================================

def category_c(duration, config, out, ns, quick=False):
    """Interbrain phase sync predicting behavior: kappa sweep.

    Uses build_synthetic_interbrain_session which creates shared EEG phase
    at 10 Hz posterior that predicts blendshape movements. Tests cross-modal
    coupling detection via interbrain features.
    """
    results = []
    kappas = ([0.3, 0.5, 0.7] if quick
              else [0.2, 0.3, 0.4, 0.5, 0.6, 0.8])

    _header('C: Interbrain Cross-Modal Sensitivity',
            f'{len(kappas)} kappas x {ns["c"]} seeds')

    jobs = [(kappa, s) for kappa in kappas for s in range(ns['c'])]
    def _build(params):
        kappa, s = params
        seed = 15000 + int(kappa * 1000) * 100 + s
        sess = build_synthetic_interbrain_session(
            duration, coupling_freq=10.0, coupling_roi='posterior',
            kappa=kappa, seed=seed)
        return _ensure_v2_features(sess, config)

    for (kappa, s), session in _prefetch_loop(jobs, _build):
        r = analyze(f'c_ib_k{kappa:.2f}_{s}', session, config,
                    expected_pathways=EXPECTED_INTERBRAIN)
        if r is None:
            continue
        r['sub'] = 'c'
        r['kappa'] = kappa
        r['modality'] = 'eeg_interbrain'
        results.append(r)
        det = 'HIT' if r['tp'] else 'miss'
        fp_n = len(r['fp'])
        fp_str = f' FP={fp_n}' if fp_n else ''
        auc_str = ''
        if r.get('temporal_auc'):
            auc_vals = list(r['temporal_auc'].values())
            if auc_vals:
                auc_str = f' AUC={max(auc_vals):.3f}'
        print(f"  IB k={kappa:.2f} s{s}: {det}{fp_str}{auc_str} ({r['elapsed_s']}s)")

    return results


# ===================================================================
# Category D: Behavioral Modality Sensitivity
# ===================================================================

def category_d(duration, config, out, ns, quick=False):
    """Same-modality coupling for BL and Pose: kappa sweep.

    Uses build_synthetic_session_v2 which injects coupling directly in
    V2 feature space. Tests whether doubly-sparse selection can identify
    coupled features in behavioral modalities.
    """
    results = []
    kappas = ([0.15, 0.30, 0.50] if quick
              else [0.10, 0.15, 0.20, 0.30, 0.50])
    test_mods = ['blendshapes_v2', 'pose_features']

    for mod in test_mods:
        _header(f'D: {MOD_LABELS[mod]} Sensitivity',
                f'{len(kappas)} kappas x {ns["d"]} seeds')

        expected = {(mod, mod)}
        jobs = [(kappa, s) for kappa in kappas for s in range(ns['d'])]

        def _build(params, _mod=mod):
            kappa, s = params
            kd = {m: 0.0 for m in BEHAVIORAL_MODS}
            kd[_mod] = kappa
            seed = (20000 + BEHAVIORAL_MODS.index(_mod) * 10000
                    + int(kappa * 1000) * 100 + s)
            return build_synthetic_session_v2(duration, kd, seed=seed)

        for (kappa, s), session in _prefetch_loop(jobs, _build):
            r = analyze(f'd_{MOD_LABELS[mod]}_k{kappa:.2f}_{s}',
                        session, config, expected_pathways=expected)
            if r is None:
                continue
            r['sub'] = 'd'
            r['kappa'] = kappa
            r['modality'] = mod
            results.append(r)
            det = 'HIT' if r['tp'] else 'miss'
            auc_str = ''
            if r.get('temporal_auc'):
                auc_vals = list(r['temporal_auc'].values())
                if auc_vals:
                    auc_str = f' AUC={max(auc_vals):.3f}'
            print(f"  {MOD_LABELS[mod]} k={kappa:.2f} s{s}: "
                  f"{det}{auc_str} ({r['elapsed_s']}s)")

    return results


# ===================================================================
# Category E: Feature Specificity
# ===================================================================

def category_e(duration, config, out, ns, quick=False):
    """Multi-frequency, multi-ROI feature recovery verification.

    For each (frequency, ROI) combination, injects EEG coupling and checks:
    1. Is the pathway detected? (sensitivity)
    2. Are the correct frequency bins selected? (feature specificity)

    This is the key test for the doubly-sparse pipeline's ability to
    identify WHICH features are coupled, not just WHETHER coupling exists.
    """
    results = []

    freqs = [5.0, 10.0, 20.0, 35.0] if not quick else [6.5, 10.0]
    rois = (['frontal', 'posterior'] if quick
            else ['frontal', 'posterior', 'left_temp', 'right_temp'])

    _header('E: Feature Specificity',
            f'{len(freqs)} freqs x {len(rois)} ROIs')

    jobs = [(freq, roi) for freq in freqs for roi in rois]

    def _build(params):
        freq, roi = params
        roi_idx = EEG_ROI_NAMES.index(roi)
        seed = 60000 + int(freq * 100) + roi_idx * 10
        sess = build_synthetic_wavelet_session(
            duration, coupling_freq=freq, coupling_roi=roi,
            kappa=0.6, seed=seed)
        return _ensure_v2_features(sess, config)

    for (freq, roi), session in _prefetch_loop(jobs, _build):
        freq_idx = int(np.argmin(np.abs(WAVELET_CENTER_FREQS - freq)))

        r = analyze(f'e_f{freq:.0f}_{roi}', session, config,
                    expected_pathways=EXPECTED_EEG,
                    expected_freq_idx=freq_idx)
        if r is None:
            continue
        r['sub'] = 'e'
        r['freq'] = freq
        r['roi'] = roi
        r['expected_freq_idx'] = freq_idx

        det = 'HIT' if r['tp'] else 'miss'
        frec = 'freq_ok' if r.get('freq_recovered') else 'freq_miss'
        print(f"  f={freq:.0f}Hz {roi:12s}: {det}, {frec} "
              f"({r['elapsed_s']}s)")
        results.append(r)

    return results


# ===================================================================
# Category F: Temporal Pattern Robustness
# ===================================================================

def category_f(duration, config, out, ns, quick=False):
    """Temporal coupling patterns at the raw EEG level.

    Tests whether the block selection + SSE rank test correctly handles
    different temporal coupling patterns (the "temporally sparse" half
    of doubly-sparse detection).
    """
    results = []
    patterns = (['sustained', 'sparse', 'late_onset'] if quick
                else ['sustained', 'sparse', 'single_event',
                      'late_onset', 'early_offset'])

    _header('F: Temporal Patterns',
            f'{len(patterns)} patterns x {ns["f"]} seeds')

    for pat in patterns:
        jobs = list(range(ns['f']))

        def _build(s, _pat=pat):
            seed = 30000 + patterns.index(_pat) * 1000 + s * 100
            return build_wavelet_custom_gate(
                duration, freq=6.5, roi='frontal', kappa=0.6,
                pattern=_pat, seed=seed, config=config)

        for s, session in _prefetch_loop(jobs, _build):
            r = analyze(f'f_{pat}_{s}', session, config,
                        expected_pathways=EXPECTED_EEG)
            if r is None:
                continue
            r['sub'] = 'f'
            r['pattern'] = pat
            results.append(r)
            det = 'HIT' if r['tp'] else 'miss'
            auc_str = ''
            if r.get('temporal_auc'):
                auc_vals = list(r['temporal_auc'].values())
                if auc_vals:
                    auc_str = f' AUC={auc_vals[0]:.3f}'
            print(f"  {pat} s{s}: {det}{auc_str} ({r['elapsed_s']}s)")

    return results


# ===================================================================
# Category G: Directionality Calibration
# ===================================================================

def category_g(duration, config, out, ns, quick=False):
    """Verify coupling detected in correct direction only.

    Builds P1->P2 wavelet coupling and tests both directions.
    Forward (p1_to_p2) should detect; reverse (p2_to_p1) should not.
    """
    results = []

    _header('G: Directionality', f'{ns["g"]} seeds')

    def _build(s):
        sess = build_synthetic_wavelet_session(
            duration, coupling_freq=6.5, coupling_roi='frontal',
            kappa=0.6, seed=40000 + s * 1000)
        return _ensure_v2_features(sess, config)

    for s, session in _prefetch_loop(range(ns['g']), _build):
        # Forward: should detect
        r_fwd = analyze(f'g_fwd_{s}', session, config,
                        direction='p1_to_p2',
                        expected_pathways=EXPECTED_EEG)
        if r_fwd is not None:
            r_fwd['sub'] = 'g_fwd'
            results.append(r_fwd)

        # Reverse: should NOT detect (no coupling in this direction)
        r_rev = analyze(f'g_rev_{s}', session, config,
                        direction='p2_to_p1',
                        expected_pathways=set())
        if r_rev is not None:
            r_rev['sub'] = 'g_rev'
            results.append(r_rev)

        if r_fwd is not None and r_rev is not None:
            fwd = 'HIT' if r_fwd['tp'] else 'miss'
            rev = 'FP!' if r_rev['fp'] else 'clean'
            print(f"  s{s}: fwd={fwd}, rev={rev}")
        else:
            print(f"  s{s}: [resumed]")

    return results


# ===================================================================
# Reporting
# ===================================================================

def report_category_a(results):
    """Null control: FPR per sub-category."""
    subs = defaultdict(list)
    for r in results:
        subs[r['sub']].append(r)

    print(f"\n{'=' * 60}")
    print("CATEGORY A: NULL CONTROL")
    print(f"{'=' * 60}")

    all_pass = True
    names = {
        'a1': 'Null (EEG)',
        'a2': 'Null (All)',
        'a3': 'Shared Trend',
        'a4': 'Corr Activity',
        'a5': 'Shared Noise',
    }

    for sub in sorted(subs):
        rs = subs[sub]
        total_fp = sum(len(r['fp']) for r in rs)
        total_pathways = sum(
            len(r['fp']) + len(r['tn']) + len(r['tp']) + len(r['fn'])
            for r in rs)
        fpr = total_fp / max(total_pathways, 1)
        status = 'PASS' if fpr <= 0.10 else 'FAIL'
        if fpr > 0.10:
            all_pass = False
        print(f"  {names.get(sub, sub):15s}: FPR={fpr:.1%} "
              f"({total_fp}/{total_pathways})  [{status}]")
        if total_fp > 0:
            for r in rs:
                for fp_name in r['fp']:
                    print(f"    {r['name']}: {fp_name}")

    return all_pass


def report_power_curve(results, category_name):
    """Sensitivity sweep: detection rate + AUC per (modality, kappa)."""
    power = defaultdict(lambda: defaultdict(list))
    auc_by_mod_kappa = defaultdict(lambda: defaultdict(list))
    for r in results:
        mod = r.get('modality', '?')
        kappa = r.get('kappa', 0)
        detected = len(r['tp']) > 0
        power[mod][kappa].append(detected)
        # Collect AUC values for detected pathways
        for auc_val in r.get('temporal_auc', {}).values():
            auc_by_mod_kappa[mod][kappa].append(auc_val)

    print(f"\n{'=' * 60}")
    print(f"CATEGORY {category_name}: SENSITIVITY SWEEP")
    print(f"{'=' * 60}")

    all_pass = True
    for mod in sorted(power):
        label = MOD_LABELS.get(mod, mod)
        print(f"\n  {label}:")
        print(f"    {'kappa':>5s}  {'Det%':>5s}  {'AUC':>7s}  {'bar':20s}  n")
        print(f"    {'-'*5}  {'-'*5}  {'-'*7}  {'-'*20}  {'-'*5}")
        kmin = None
        for kappa in sorted(power[mod]):
            hits = power[mod][kappa]
            rate = sum(hits) / len(hits)
            bar = '#' * int(rate * 20) + '.' * (20 - int(rate * 20))
            # Per-kappa AUC
            kappa_aucs = auc_by_mod_kappa[mod][kappa]
            if kappa_aucs:
                auc_str = f'{np.mean(kappa_aucs):.3f}'
            else:
                auc_str = '   -  '
            print(f"    {kappa:5.2f}  {rate:5.0%}  {auc_str:>7s}  "
                  f"[{bar}]  {sum(hits)}/{len(hits)}")
            if rate >= 0.80 and kmin is None:
                kmin = kappa
        if kmin is not None:
            print(f"    -> k_min(80% power) = {kmin:.2f}")
        else:
            print(f"    -> Never reached 80% power")
            all_pass = False

    # Frequency recovery summary (Category B only)
    freq_results = [r for r in results if r.get('freq_recovered') is not None]
    if freq_results:
        freq_ok = sum(1 for r in freq_results
                      if r['freq_recovered'] and r['tp'])
        freq_det = sum(1 for r in freq_results if r['tp'])
        if freq_det > 0:
            print(f"\n  Frequency recovery: {freq_ok}/{freq_det} "
                  f"({freq_ok/freq_det:.0%} of detected)")

    # Overall AUC summary
    all_auc = []
    for r in results:
        all_auc.extend(r.get('temporal_auc', {}).values())
    if all_auc:
        print(f"  Overall temporal AUC: mean={np.mean(all_auc):.3f} "
              f"[{min(all_auc):.3f}, {max(all_auc):.3f}]")

    return all_pass


def report_category_e(results):
    """Feature specificity: detection rate + frequency recovery rate + AUC."""
    print(f"\n{'=' * 60}")
    print("CATEGORY E: FEATURE SPECIFICITY")
    print(f"{'=' * 60}")

    detected = sum(1 for r in results if r['tp'])
    freq_ok = sum(1 for r in results if r.get('freq_recovered') is True)
    total = len(results)

    for r in results:
        freq = r.get('freq', 0)
        roi = r.get('roi', '?')
        det = 'HIT' if r['tp'] else 'miss'
        frec = 'freq_ok' if r.get('freq_recovered') is True else 'freq_miss'
        aucs = list(r.get('temporal_auc', {}).values())
        auc_str = f' AUC={max(aucs):.3f}' if aucs else ''
        print(f"  f={freq:.0f}Hz {roi:12s}: {det}, {frec}{auc_str}")

    det_rate = detected / max(total, 1)
    freq_rate = freq_ok / max(total, 1)
    print(f"\n  Detection: {detected}/{total} ({det_rate:.0%})")
    print(f"  Freq recovery: {freq_ok}/{total} ({freq_rate:.0%})")

    all_aucs = []
    for r in results:
        all_aucs.extend(r.get('temporal_auc', {}).values())
    if all_aucs:
        print(f"  Temporal AUC: mean={np.mean(all_aucs):.3f} "
              f"[{min(all_aucs):.3f}, {max(all_aucs):.3f}]")

    return det_rate >= 0.50


def report_category_f(results):
    """Temporal patterns: detection rate + AUC per pattern."""
    by_pat = defaultdict(list)
    for r in results:
        by_pat[r.get('pattern', '?')].append(r)

    print(f"\n{'=' * 60}")
    print("CATEGORY F: TEMPORAL PATTERNS")
    print(f"{'=' * 60}")
    print(f"  {'Pattern':15s}  {'Det':>5s}  {'AUC mean':>8s}  "
          f"{'AUC range':>13s}  Status")
    print(f"  {'-'*15}  {'-'*5}  {'-'*8}  {'-'*13}  {'-'*6}")

    all_pass = True
    for pat in ['sustained', 'sparse', 'single_event',
                'late_onset', 'early_offset']:
        if pat not in by_pat:
            continue
        rs = by_pat[pat]
        det = sum(1 for r in rs if r['tp'])
        rate = det / len(rs) if rs else 0
        status = 'PASS' if rate >= 0.50 else 'FAIL'
        if rate < 0.50:
            all_pass = False
        aucs = []
        for r in rs:
            aucs.extend(r.get('temporal_auc', {}).values())
        if aucs:
            auc_mean = f'{np.mean(aucs):.3f}'
            auc_range = f'[{min(aucs):.3f},{max(aucs):.3f}]'
        else:
            auc_mean = '   -   '
            auc_range = '      -      '
        print(f"  {pat:15s}  {rate:5.0%}  {auc_mean:>8s}  "
              f"{auc_range:>13s}  {status}")

    return all_pass


def report_category_g(results):
    """Directionality: forward detection, reverse FP control."""
    print(f"\n{'=' * 60}")
    print("CATEGORY G: DIRECTIONALITY")
    print(f"{'=' * 60}")

    subs = defaultdict(list)
    for r in results:
        subs[r['sub']].append(r)

    all_pass = True

    fwd = subs.get('g_fwd', [])
    rev = subs.get('g_rev', [])

    fwd_det = sum(1 for r in fwd if r['tp'])
    print(f"  Forward:  {fwd_det}/{len(fwd)} detected")

    rev_fp = sum(len(r['fp']) for r in rev)
    status = 'PASS' if rev_fp <= 1 else 'FAIL'
    if rev_fp > 1:
        all_pass = False
    print(f"  Reverse:  {rev_fp} FPs [{status}]")

    return all_pass


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CADENCE V2 Validation Battery')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (300s, 1 seed)')
    parser.add_argument('--full', action='store_true',
                        help='Full battery (1800s, many seeds)')
    parser.add_argument('--category', nargs='*', default=None,
                        help='Run only these categories (a b c d e f g)')
    parser.add_argument('--output', default='results/validation_battery')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip', default=None,
                        help='Comma-separated subcategories to skip')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing sessions.jsonl')
    args = parser.parse_args()

    config = load_config(args.config)
    config['pipeline'] = 'v2'
    if args.device:
        config['device'] = args.device

    os.makedirs(args.output, exist_ok=True)

    # Seed counts and durations
    if args.quick:
        dur = 300
        ns = {'a1': 2, 'a2': 2, 'a3': 1, 'a4': 1, 'a5': 1,
              'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}
    elif args.full:
        dur = 1800
        ns = {'a1': 10, 'a2': 10, 'a3': 5, 'a4': 5, 'a5': 5,
              'b': 5, 'c': 5, 'd': 3, 'e': 1, 'f': 2, 'g': 3}
    else:
        # Default: moderate
        dur = 1800
        ns = {'a1': 3, 'a2': 3, 'a3': 2, 'a4': 2, 'a5': 2,
              'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 1, 'g': 2}

    cats_to_run = set()
    if args.category:
        cats_to_run = set(c.lower() for c in args.category)
    else:
        cats_to_run = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    print(f"CADENCE V2 Validation Battery")
    print(f"  Mode: {'quick' if args.quick else 'full' if args.full else 'default'}")
    print(f"  Duration: {dur}s")
    print(f"  Categories: {sorted(cats_to_run)}")
    print(f"  Device: {config['device']}")

    # Set up incremental saving and resume
    global _INCREMENTAL_PATH, _COMPLETED_SESSIONS, _SKIP_SUBS
    _INCREMENTAL_PATH = os.path.join(args.output, 'sessions.jsonl')

    if args.skip:
        _SKIP_SUBS = set(s.strip().lower() for s in args.skip.split(','))
        print(f"  Skipping: {sorted(_SKIP_SUBS)}")

    if args.resume and os.path.exists(_INCREMENTAL_PATH):
        with open(_INCREMENTAL_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        _COMPLETED_SESSIONS.add(d['name'])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  Resuming: {len(_COMPLETED_SESSIONS)} sessions done")

    all_results = {}
    verdicts = {}
    t0_all = time.time()

    try:
        if 'a' in cats_to_run:
            all_results['a'] = category_a(
                dur, config, args.output, ns, args.quick)
            verdicts['A: Null Control'] = report_category_a(
                all_results['a'])
            _save_category('a', all_results['a'], args.output)

        if 'b' in cats_to_run:
            all_results['b'] = category_b(
                dur, config, args.output, ns, args.quick)
            verdicts['B: EEG Sensitivity'] = report_power_curve(
                all_results['b'], 'B: EEG Sensitivity')
            _save_category('b', all_results['b'], args.output)

        if 'c' in cats_to_run:
            all_results['c'] = category_c(
                dur, config, args.output, ns, args.quick)
            verdicts['C: Interbrain'] = report_power_curve(
                all_results['c'], 'C: Interbrain')
            _save_category('c', all_results['c'], args.output)

        if 'd' in cats_to_run:
            all_results['d'] = category_d(
                dur, config, args.output, ns, args.quick)
            verdicts['D: Behavioral'] = report_power_curve(
                all_results['d'], 'D: Behavioral')
            _save_category('d', all_results['d'], args.output)

        if 'e' in cats_to_run:
            all_results['e'] = category_e(
                dur, config, args.output, ns, args.quick)
            verdicts['E: Feature Spec'] = report_category_e(
                all_results['e'])
            _save_category('e', all_results['e'], args.output)

        if 'f' in cats_to_run:
            all_results['f'] = category_f(
                dur, config, args.output, ns, args.quick)
            verdicts['F: Temporal'] = report_category_f(
                all_results['f'])
            _save_category('f', all_results['f'], args.output)

        if 'g' in cats_to_run:
            all_results['g'] = category_g(
                dur, config, args.output, ns, args.quick)
            verdicts['G: Directionality'] = report_category_g(
                all_results['g'])
            _save_category('g', all_results['g'], args.output)

    except Exception:
        traceback.print_exc()

    # Final summary
    elapsed_total = time.time() - t0_all
    print(f"\n\n{'=' * 60}")
    print(f"VALIDATION BATTERY SUMMARY")
    print(f"{'=' * 60}")
    for cat, passed in verdicts.items():
        print(f"  {cat:25s}  {'PASS' if passed else '** FAIL **'}")
    n_pass = sum(verdicts.values())
    n_total = len(verdicts)
    print(f"\n  Overall: {n_pass}/{n_total} categories passed")
    print(f"  Total time: {elapsed_total / 60:.1f} min")

    # Save results
    save_data = {
        'verdicts': {k: bool(v) for k, v in verdicts.items()},
        'mode': 'quick' if args.quick else 'full' if args.full else 'default',
        'duration': dur,
        'elapsed_min': round(elapsed_total / 60, 1),
    }
    for cat, rs in all_results.items():
        save_data[f'category_{cat}'] = [{
            k: v for k, v in r.items()
            if k not in ('pathway_details', 'feature_info')
        } for r in rs]

    with open(os.path.join(args.output, 'battery_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
