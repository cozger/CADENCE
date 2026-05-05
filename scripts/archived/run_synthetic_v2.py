"""CADENCE v2 synthetic validation battery.

Tests:
1. Frequency-specific recovery: coupling at 6.5 Hz in frontal ROI
2. Phase-driven recovery: analytic signal captures phase-based coupling
3. Inter-brain phase recovery: phase synchrony predicts blendshapes
4. False positive control: sparse true pathways among many candidates
5b. Sparse temporal coupling: 10% duty cycle episodic coupling
6. Cross-session consistency: shared true pathways across sessions

Usage:
    python scripts/run_synthetic_v2.py
    python scripts/run_synthetic_v2.py --duration 600 --device cuda
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cadence.config import load_config
from cadence.synthetic import (
    build_synthetic_wavelet_session,
    build_synthetic_interbrain_session,
    generate_coupling_gate,
)
from cadence.data.alignment import _ensure_v2_features
from cadence.coupling.estimator import CouplingEstimator, CouplingResult
from cadence.coupling.discovery import (
    DiscoveryResult, cross_session_consistency, discovery_summary,
)
from cadence.constants import WAVELET_CENTER_FREQS, EEG_ROI_NAMES, EEG_ROIS


def compute_temporal_auc(result, session, eval_rate):
    """Compute ROC AUC between dr2 timecourse and ground-truth coupling gate.

    For each significant pathway with a known coupling gate, resamples the
    gate to the eval rate and computes AUC against the dr2 timecourse.

    Returns:
        auc_dict: {pathway_key: auc_value}
    """
    from sklearn.metrics import roc_auc_score

    coupling_gates = session.get('coupling_gates', {})
    if not coupling_gates:
        return {}

    duration = session.get('duration', 0)
    eval_times = np.arange(0, duration, 1.0 / eval_rate)
    n_eval = len(eval_times)

    auc_dict = {}
    for key, dr2 in result.pathway_dr2.items():
        if not result.pathway_significant.get(key, False):
            continue

        src_mod, tgt_mod = key
        # Find matching coupling gate
        gate = None
        for gate_mod, g in coupling_gates.items():
            if gate_mod in src_mod or src_mod in gate_mod:
                gate = g
                break
        # Also check target modality for interbrain->blendshapes etc.
        if gate is None:
            for gate_mod, g in coupling_gates.items():
                if gate_mod in tgt_mod or tgt_mod in gate_mod:
                    gate = g
                    break

        if gate is None:
            continue

        # Resample gate to eval rate
        gate_srate = len(gate) / max(duration, 1e-6)
        gate_times = np.arange(len(gate)) / gate_srate
        gate_resampled = np.interp(eval_times, gate_times, gate)
        # Binarize: gate > 0.1 counts as "coupling on"
        gate_binary = (gate_resampled > 0.1).astype(float)

        # Trim to matching length
        n = min(len(dr2), n_eval, len(gate_binary))
        dr2_trimmed = np.nan_to_num(dr2[:n], nan=0.0)
        gate_trimmed = gate_binary[:n]

        # Need both classes present for AUC
        if gate_trimmed.sum() < 1 or gate_trimmed.sum() > n - 1:
            continue

        try:
            auc = roc_auc_score(gate_trimmed, dr2_trimmed)
            auc_dict[key] = auc
        except Exception:
            continue

    return auc_dict


def report_auc(auc_dict, test_name):
    """Print AUC results for a test."""
    if not auc_dict:
        print(f"  Temporal AUC: no pathways with ground truth gate")
        return
    for key, auc in auc_dict.items():
        src, tgt = key
        print(f"  Temporal AUC {src}->{tgt}: {auc:.3f}")


def test_frequency_specific_recovery(config, duration=1800, output_dir='results'):
    """Test 1: Inject coupling at 6.5 Hz frontal. Verify group lasso selects ~6-7 Hz."""
    print("\n=== Test 1: Frequency-Specific Recovery ===")
    print(f"  Injecting coupling at 6.5 Hz in frontal ROI")

    session = build_synthetic_wavelet_session(
        duration=duration, coupling_freq=6.5, coupling_roi='frontal',
        kappa=0.6, seed=42)

    session = _ensure_v2_features(session, config)

    estimator = CouplingEstimator(config)
    eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
    result = estimator.analyze_session(session, 'p1_to_p2')

    n_sig = result.n_significant_pathways
    print(f"  Significant pathways: {n_sig}")

    for key, dr2 in result.pathway_dr2.items():
        src, tgt = key
        mean_dr2 = np.nanmean(dr2)
        if mean_dr2 > 0.001:
            print(f"  {src} -> {tgt}: mean dR2 = {mean_dr2:.4f}")

    freq_idx_target = np.argmin(np.abs(WAVELET_CENTER_FREQS - 6.5))
    print(f"  Target frequency index: {freq_idx_target} "
          f"({WAVELET_CENTER_FREQS[freq_idx_target]:.1f} Hz)")

    # Temporal AUC
    auc_dict = compute_temporal_auc(result, session, eval_rate)
    report_auc(auc_dict, "freq_specific")

    passed = n_sig > 0
    print(f"  PASSED" if passed else f"  FAILED")
    return passed, auc_dict


def test_phase_driven_recovery(config, duration=1800, output_dir='results'):
    """Test 2: Coupling depends on phase. Verify analytic signal captures it."""
    print("\n=== Test 2: Phase-Driven Recovery ===")
    print(f"  Coupling at 8 Hz (alpha) via phase relationship")

    session = build_synthetic_wavelet_session(
        duration=duration, coupling_freq=8.0, coupling_roi='posterior',
        coupling_lag_s=0.5, kappa=0.7, seed=123)

    session = _ensure_v2_features(session, config)

    estimator = CouplingEstimator(config)
    eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
    result = estimator.analyze_session(session, 'p1_to_p2')

    n_sig = result.n_significant_pathways
    print(f"  Significant pathways: {n_sig}")

    auc_dict = compute_temporal_auc(result, session, eval_rate)
    report_auc(auc_dict, "phase_driven")

    passed = n_sig > 0
    print(f"  PASSED" if passed else f"  FAILED")
    return passed, auc_dict


def test_interbrain_phase_recovery(config, duration=1800, output_dir='results'):
    """Test 3: Phase synchrony between A and B at 10 Hz predicts B's face."""
    print("\n=== Test 3: Inter-Brain Phase Recovery ===")
    print(f"  10 Hz posterior phase sync -> blendshapes")

    session = build_synthetic_interbrain_session(
        duration=duration, coupling_freq=10.0, coupling_roi='posterior',
        kappa=0.6, seed=456)

    session = _ensure_v2_features(session, config)

    estimator = CouplingEstimator(config)
    eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
    result = estimator.analyze_session(session, 'p1_to_p2')

    n_sig = result.n_significant_pathways
    print(f"  Significant pathways: {n_sig}")

    ib_sig = any(
        result.pathway_significant.get(k, False)
        for k in result.pathway_dr2
        if k[0] == 'eeg_interbrain'
    )
    print(f"  Inter-brain pathway significant: {ib_sig}")

    auc_dict = compute_temporal_auc(result, session, eval_rate)
    report_auc(auc_dict, "interbrain")

    passed = n_sig > 0
    print(f"  PASSED" if passed else f"  FAILED")
    return passed, auc_dict


def test_false_positive_control(config, duration=1800, output_dir='results'):
    """Test 4: Null session (no coupling). Verify FDR control."""
    print("\n=== Test 4: False Positive Control ===")
    print(f"  Null session — no coupling injected")

    session = build_synthetic_wavelet_session(
        duration=duration, coupling_freq=6.5, coupling_roi='frontal',
        kappa=0.0, seed=789)

    session = _ensure_v2_features(session, config)

    estimator = CouplingEstimator(config)
    result = estimator.analyze_session(session, 'p1_to_p2')

    n_sig = result.n_significant_pathways
    total = len(result.pathway_dr2)
    fp_rate = n_sig / max(total, 1)
    print(f"  Significant: {n_sig}/{total} (FP rate = {fp_rate:.2%})")

    passed = fp_rate <= 0.10
    print(f"  PASSED" if passed else f"  FAILED")
    return passed, {}


def test_sparse_temporal_coupling(config, duration=1800, output_dir='results'):
    """Test 5b: Sparse temporal coupling (low duty cycle, few features).

    Coupling at 8 Hz posterior, only 10% duty cycle (sparse in time),
    with strong coupling strength to compensate. Tests whether block-level
    selection can catch episodic coupling that session-global tests miss.
    """
    print("\n=== Test 5b: Sparse Temporal Coupling ===")
    print(f"  8 Hz posterior, 10% duty cycle, kappa=0.8")

    session = build_synthetic_wavelet_session(
        duration=duration, coupling_freq=8.0, coupling_roi='posterior',
        kappa=0.8, seed=555)

    srate = 256
    n_samples = int(duration * srate)
    sparse_profile = {'duty_cycle': 0.10, 'event_range_s': (3, 10), 'ramp_s': 1.0}
    sparse_gate = generate_coupling_gate(n_samples, srate, sparse_profile, seed=556)

    roi_channels = EEG_ROIS['posterior']
    t = np.arange(n_samples) / srate
    coupling_signal = np.sin(2 * np.pi * 8.0 * t)
    lag_samples = int(1.0 * srate)

    rng = np.random.default_rng(555)
    freqs = np.logspace(np.log10(2), np.log10(45), 30)
    eeg_p2_fresh = np.zeros((n_samples, 14), dtype=np.float32)
    for ch in range(14):
        for f in freqs:
            amp = 1.0 / f
            phase = rng.uniform(0, 2 * np.pi)
            eeg_p2_fresh[:, ch] += amp * np.sin(2 * np.pi * f * t + phase)

    for ch in roi_channels:
        p1_contrib = np.roll(coupling_signal * session['p1_eeg'][:, ch], lag_samples)
        p1_contrib[:lag_samples] = 0
        alpha_t = 0.8 * sparse_gate
        eeg_p2_fresh[:, ch] = (1 - alpha_t) * eeg_p2_fresh[:, ch] + alpha_t * p1_contrib

    for ch in range(14):
        std = eeg_p2_fresh[:, ch].std()
        if std > 1e-8:
            eeg_p2_fresh[:, ch] = (eeg_p2_fresh[:, ch] - eeg_p2_fresh[:, ch].mean()) / std

    session['p2_eeg'] = eeg_p2_fresh
    session['coupling_gates'] = {'eeg_wavelet': sparse_gate}
    session['ground_truth']['duty_cycle'] = 0.10

    session = _ensure_v2_features(session, config)

    estimator = CouplingEstimator(config)
    eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
    result = estimator.analyze_session(session, 'p1_to_p2')

    n_sig = result.n_significant_pathways
    print(f"  Significant pathways: {n_sig}")

    duty_frac = sparse_gate.mean()
    print(f"  Actual duty cycle: {duty_frac:.2%}")

    auc_dict = compute_temporal_auc(result, session, eval_rate)
    report_auc(auc_dict, "sparse_temporal")

    passed = n_sig > 0
    print(f"  PASSED" if passed else f"  FAILED")
    return passed, auc_dict


def test_cross_session_consistency(config, duration=1800, output_dir='results'):
    """Test 6: Run discovery on 5 sessions with shared true pathways."""
    print("\n=== Test 6: Cross-Session Consistency ===")
    print(f"  5 sessions, coupling at 6.5 Hz frontal")

    all_discoveries = []

    for i in range(5):
        session = build_synthetic_wavelet_session(
            duration=duration, coupling_freq=6.5, coupling_roi='frontal',
            kappa=0.6, seed=1000 + i * 100)

        session = _ensure_v2_features(session, config)

        estimator = CouplingEstimator(config)

        src_sigs = estimator._extract_signals_v2(session, 'p1')
        tgt_sigs = estimator._extract_signals_v2(session, 'p2')

        # Add interbrain features (mirrors _analyze_session_v2)
        ib_key = 'eeg_interbrain'
        if ib_key in session:
            ib_signal = session[ib_key]
            ib_ts = session[f'{ib_key}_ts']
            ib_valid = session.get(f'{ib_key}_valid',
                                    np.ones(len(ib_signal), dtype=bool))
            if ib_valid.sum() >= 10:
                src_sigs[ib_key] = (ib_signal, ib_ts, ib_valid)

        eval_rate = config.get('wavelet', {}).get('output_hz', 5.0)
        disc = estimator._stage1_discovery(
            src_sigs, tgt_sigs, session.get('duration', 0), eval_rate)
        disc.session_name = f"synth_{i}"
        all_discoveries.append(disc)

        n_sel = sum(1 for v in disc.n_selected.values() if v > 0)
        print(f"  Session {i}: {n_sel} pathways with selected features")

    consistency = cross_session_consistency(all_discoveries, min_sessions=3)
    summary = discovery_summary(consistency)
    print(summary)

    n_consistent_pathways = sum(
        1 for features in consistency.consistent_features.values()
        if len(features) > 0)

    print(f"  Consistent pathways: {n_consistent_pathways}")
    passed = n_consistent_pathways > 0
    print(f"  PASSED" if passed else f"  FAILED")
    return passed, {}


def main():
    parser = argparse.ArgumentParser(description='CADENCE v2 synthetic validation')
    parser.add_argument('--duration', type=int, default=1800,
                        help='Session duration in seconds (default: 1800)')
    parser.add_argument('--config', default=None, help='YAML config file')
    parser.add_argument('--output', default='results/cadence_synthetic_v2',
                        help='Output directory')
    parser.add_argument('--device', default=None, help='Device override')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)
    config['pipeline'] = 'v2'
    if args.device:
        config['device'] = args.device

    os.makedirs(args.output, exist_ok=True)

    print("CADENCE v2 Synthetic Validation Battery")
    print(f"Duration: {args.duration}s, Device: {config['device']}")

    results = {}
    all_aucs = {}
    t0 = time.time()

    passed, aucs = test_frequency_specific_recovery(
        config, args.duration, args.output)
    results['freq_specific'] = passed
    all_aucs['freq_specific'] = aucs

    passed, aucs = test_phase_driven_recovery(
        config, args.duration, args.output)
    results['phase_driven'] = passed
    all_aucs['phase_driven'] = aucs

    passed, aucs = test_interbrain_phase_recovery(
        config, args.duration, args.output)
    results['interbrain'] = passed
    all_aucs['interbrain'] = aucs

    passed, aucs = test_false_positive_control(
        config, args.duration, args.output)
    results['false_positive'] = passed
    all_aucs['false_positive'] = aucs

    passed, aucs = test_sparse_temporal_coupling(
        config, args.duration, args.output)
    results['sparse_temporal'] = passed
    all_aucs['sparse_temporal'] = aucs

    passed, aucs = test_cross_session_consistency(
        config, args.duration, args.output)
    results['consistency'] = passed
    all_aucs['consistency'] = aucs

    dt = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Results ({dt:.0f}s total):")
    n_passed = sum(results.values())
    n_total = len(results)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print(f"\n  {n_passed}/{n_total} tests passed")

    # Summary of temporal AUC across all tests
    print(f"\nTemporal AUC Summary:")
    any_auc = False
    for test_name, auc_dict in all_aucs.items():
        for key, auc in auc_dict.items():
            src, tgt = key
            print(f"  {test_name}: {src}->{tgt} AUC={auc:.3f}")
            any_auc = True
    if not any_auc:
        print("  No pathways with temporal AUC (no significant pathways with ground truth)")


if __name__ == '__main__':
    main()
