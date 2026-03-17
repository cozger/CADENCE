"""CADENCE synthetic validation with timing accuracy and visualizations.

Tests on 3000s Lorenz sessions with realistic coupling parameters.
Reports:
  - Detection accuracy (TP/FP/TN/FN per modality)
  - Timing accuracy via AUC-ROC (dR2 vs ground-truth coupling gate)
  - Visualizations: coupling gate overlay, ROC curves, summary heatmap

Usage:
    python scripts/run_synthetic_timing.py
    python scripts/run_synthetic_timing.py --duration 1800 --n-seeds 3
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cadence.config import load_config
from cadence.synthetic import build_synthetic_session_permod
from cadence.coupling.estimator import CouplingEstimator
from cadence.constants import (
    MODALITY_ORDER, MOD_SHORT, MODALITY_COLORS,
    COUPLING_PROFILES, SYNTH_MODALITY_CONFIG,
)


# ---------------------------------------------------------------------------
# Test definitions — realistic coupling parameters
# ---------------------------------------------------------------------------
# kappa=0.5 is moderate coupling (not the kappa=0.7 of quick tests)
# duty cycles from COUPLING_PROFILES (modality-realistic)
TIMING_TESTS = [
    ('A_eeg_only',  {'eeg_features': 0.5, 'ecg_features': 0.0,
                     'blendshapes': 0.0, 'pose_features': 0.0},
     None, ['eeg_features']),
    ('B_bl_only',   {'eeg_features': 0.0, 'ecg_features': 0.0,
                     'blendshapes': 0.5, 'pose_features': 0.0},
     None, ['blendshapes']),
    ('C_ecg_only',  {'eeg_features': 0.0, 'ecg_features': 0.5,
                     'blendshapes': 0.0, 'pose_features': 0.0},
     None, ['ecg_features']),
    ('D_pose_only', {'eeg_features': 0.0, 'ecg_features': 0.0,
                     'blendshapes': 0.0, 'pose_features': 0.5},
     None, ['pose_features']),
    ('E_eeg_bl',    {'eeg_features': 0.5, 'ecg_features': 0.0,
                     'blendshapes': 0.5, 'pose_features': 0.0},
     None, ['eeg_features', 'blendshapes']),
    ('F_null',      {'eeg_features': 0.0, 'ecg_features': 0.0,
                     'blendshapes': 0.0, 'pose_features': 0.0},
     None, []),
]


# ---------------------------------------------------------------------------
# Timing accuracy — AUC-ROC of dR2 vs ground-truth gate
# ---------------------------------------------------------------------------
def compute_timing_aucroc(result, session, mod, gate_threshold=0.3,
                          smooth_seconds=5.0):
    """Compute AUC-ROC: how well does the dR2 timecourse localize coupling?

    Resamples ground-truth gate to the pathway's eval rate, creates binary
    labels (gate > threshold), scores = smoothed same-modality dR2.

    Returns:
        (auc, n_positive, n_negative, labels, scores) or
        (None, 0, 0, [], []) if insufficient data.
    """
    from sklearn.metrics import roc_auc_score

    gate = session.get('coupling_gates', {}).get(mod)
    if gate is None:
        return None, 0, 0, [], []

    key = (mod, mod)
    if key not in result.pathway_dr2:
        return None, 0, 0, [], []

    dr2 = result.pathway_dr2[key]
    # Use per-pathway times (may differ from result.times if per-modality rates)
    times = result.pathway_times.get(key, result.times)

    # Resample gate to pathway's eval rate
    hz = SYNTH_MODALITY_CONFIG[mod]['hz']
    gate_times = np.arange(len(gate)) / hz
    gate_resampled = np.interp(times, gate_times, gate)

    # Binary labels: coupled windows
    labels = (gate_resampled > gate_threshold).astype(int)

    # Scores: smoothed dR2 (smooth over ~5s, adapted to output rate)
    scores = np.nan_to_num(dr2, nan=0.0)
    if smooth_seconds > 0 and len(times) > 1:
        dt = times[1] - times[0]
        sigma_samples = max(1, smooth_seconds / dt)
        scores = gaussian_filter1d(scores, sigma_samples)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, n_pos, n_neg, labels.tolist(), scores.tolist()

    auc = roc_auc_score(labels, scores)
    return auc, n_pos, n_neg, labels.tolist(), scores.tolist()


def compute_timing_correlation(result, session, mod, smooth_seconds=5.0):
    """Pearson correlation between smoothed dR2 and ground-truth gate.

    Returns:
        (pearson_r, p_value) or (None, None).
    """
    from scipy.stats import pearsonr

    gate = session.get('coupling_gates', {}).get(mod)
    if gate is None:
        return None, None

    key = (mod, mod)
    if key not in result.pathway_dr2:
        return None, None

    dr2 = result.pathway_dr2[key]
    times = result.pathway_times.get(key, result.times)

    hz = SYNTH_MODALITY_CONFIG[mod]['hz']
    gate_times = np.arange(len(gate)) / hz
    gate_resampled = np.interp(times, gate_times, gate)

    scores = np.nan_to_num(dr2, nan=0.0)
    if smooth_seconds > 0 and len(times) > 1:
        dt = times[1] - times[0]
        sigma_samples = max(1, smooth_seconds / dt)
        scores = gaussian_filter1d(scores, sigma_samples)

    # Skip early/late boundary effects (first/last 30s)
    if len(times) > 1:
        dt = times[1] - times[0]
        margin = int(30 / dt)
    else:
        margin = 0
    if len(scores) > 2 * margin and margin > 0:
        gate_trim = gate_resampled[margin:-margin]
        scores_trim = scores[margin:-margin]
    else:
        gate_trim = gate_resampled
        scores_trim = scores

    if np.std(gate_trim) < 1e-10 or np.std(scores_trim) < 1e-10:
        return None, None

    r, p = pearsonr(gate_trim, scores_trim)
    return r, p


# ---------------------------------------------------------------------------
# Visualization: coupling gate overlay with dR2
# ---------------------------------------------------------------------------
def plot_gate_overlay(result, session, output_dir, test_name):
    """Plot ground-truth coupling gate overlaid with detected dR2.

    One subplot per coupled modality. Shows temporal alignment quality.
    """
    coupled_mods = [m for m in MODALITY_ORDER
                    if m in session.get('coupling_gates', {})]
    if not coupled_mods:
        return None

    fig, axes = plt.subplots(len(coupled_mods), 1,
                              figsize=(18, 3.5 * len(coupled_mods)),
                              squeeze=False)

    for i, mod in enumerate(coupled_mods):
        ax = axes[i, 0]
        gate = session['coupling_gates'][mod]
        hz = SYNTH_MODALITY_CONFIG[mod]['hz']
        gate_times = np.arange(len(gate)) / hz / 60.0

        # Ground-truth gate (background)
        ax.fill_between(gate_times, 0, gate, color='#4CAF50', alpha=0.2,
                        label='Coupling gate (truth)')
        ax.plot(gate_times, gate, color='#4CAF50', alpha=0.5, linewidth=0.8)

        # Detected dR2 timecourse (use per-pathway times)
        key = (mod, mod)
        if key in result.pathway_dr2:
            dr2 = np.nan_to_num(result.pathway_dr2[key], nan=0.0)
            pw_times = result.pathway_times.get(key, result.times)
            pw_times_min = pw_times / 60.0

            # Normalize dR2 to [0, 1] range for visual comparison
            dr2_max = max(np.percentile(dr2, 99), 1e-6)
            dr2_norm = np.clip(dr2 / dr2_max, 0, 1.5)

            color = MODALITY_COLORS.get(mod, '#1976D2')
            ax.plot(pw_times_min, dr2_norm, color=color, linewidth=1.2,
                    label=f'dR2 (normalized)', alpha=0.9)

            # Smoothed version (adapt sigma to output rate)
            dt = pw_times[1] - pw_times[0] if len(pw_times) > 1 else 0.5
            sigma_samples = max(1, 10.0 / dt)  # ~10s smoothing
            dr2_smooth = gaussian_filter1d(dr2_norm, sigma_samples)
            ax.plot(pw_times_min, dr2_smooth, color=color, linewidth=2.5,
                    alpha=0.6, linestyle='--', label='dR2 (smoothed)')

        is_sig = result.pathway_significant.get(key, False)
        sig_str = 'DETECTED' if is_sig else 'not detected'
        ax.set_title(f'{MOD_SHORT[mod]} -- {sig_str}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized amplitude')
        ax.legend(loc='upper right', fontsize=9)
        max_time = max(gate_times[-1], result.times[-1] / 60.0) if len(result.times) > 0 else 50
        ax.set_xlim(0, max_time)

    axes[-1, 0].set_xlabel('Time (min)')
    fig.suptitle(f'CADENCE Timing Accuracy -- {test_name}', fontsize=14, y=1.01)
    fig.tight_layout()

    path = os.path.join(output_dir, f'{test_name}_gate_overlay.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_roc_curves(auc_results, output_dir, test_name):
    """Plot ROC curves for each coupled modality."""
    from sklearn.metrics import roc_curve

    mods_with_auc = [(mod, data) for mod, data in auc_results.items()
                     if data['auc'] is not None]
    if not mods_with_auc:
        return None

    fig, axes = plt.subplots(1, len(mods_with_auc),
                              figsize=(5 * len(mods_with_auc), 5),
                              squeeze=False)

    for i, (mod, data) in enumerate(mods_with_auc):
        ax = axes[0, i]
        labels = np.array(data['labels'])
        scores = np.array(data['scores'])
        fpr, tpr, _ = roc_curve(labels, scores)

        color = MODALITY_COLORS.get(mod, '#1976D2')
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'AUC = {data["auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{MOD_SHORT[mod]}', fontsize=12)
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

    fig.suptitle(f'ROC Curves (Timing Accuracy) -- {test_name}', fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, f'{test_name}_roc.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_summary_heatmap(all_results, output_dir):
    """Summary heatmap: tests x modalities, colored by AUC-ROC."""
    test_names = [r['test'] for r in all_results]
    mod_names = [MOD_SHORT[m] for m in MODALITY_ORDER]

    # Build matrix: AUC values (NaN where not applicable)
    matrix = np.full((len(test_names), len(MODALITY_ORDER)), np.nan)
    for i, r in enumerate(all_results):
        for j, mod in enumerate(MODALITY_ORDER):
            auc_data = r.get('auc_results', {}).get(mod, {})
            auc = auc_data.get('auc')
            if auc is not None:
                matrix[i, j] = auc

    fig, ax = plt.subplots(figsize=(8, max(4, len(test_names) * 0.7)))

    # Custom colormap: red (0.5) -> yellow (0.75) -> green (1.0)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('auc',
        [(0.0, '#D32F2F'), (0.5, '#FFC107'), (1.0, '#4CAF50')])

    im = ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=1.0, aspect='auto')

    # Annotations
    for i in range(len(test_names)):
        for j in range(len(MODALITY_ORDER)):
            val = matrix[i, j]
            if np.isnan(val):
                # Show detection status instead
                detected = all_results[i].get('detected', {}).get(
                    MODALITY_ORDER[j], False)
                expected = MODALITY_ORDER[j] in all_results[i].get(
                    'expected_coupled', [])
                if expected:
                    txt = 'TP' if detected else 'FN'
                    color = '#4CAF50' if detected else '#D32F2F'
                elif detected:
                    txt = 'FP'
                    color = '#D32F2F'
                else:
                    txt = 'TN'
                    color = '#9E9E9E'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)
            else:
                txt = f'{val:.2f}'
                text_color = 'white' if val < 0.65 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=10, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(mod_names)))
    ax.set_xticklabels(mod_names, fontsize=11)
    ax.set_yticks(range(len(test_names)))
    ax.set_yticklabels(test_names, fontsize=10)
    ax.set_xlabel('Modality')
    ax.set_title('CADENCE Synthetic Validation: AUC-ROC (timing) + Detection',
                 fontsize=13, pad=12)

    plt.colorbar(im, ax=ax, label='AUC-ROC', shrink=0.8)
    fig.tight_layout()

    path = os.path.join(output_dir, 'timing_summary_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_detection_summary(all_results, output_dir):
    """Bar chart: detection dR2 per test with pass/fail indicators."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: dR2 values
    ax = axes[0]
    test_names = [r['test'] for r in all_results]
    bar_width = 0.2
    x = np.arange(len(test_names))

    for j, mod in enumerate(MODALITY_ORDER):
        dr2_vals = []
        colors_alpha = []
        for r in all_results:
            dr2 = r.get('dr2_values', {}).get(mod, 0.0)
            dr2_vals.append(dr2)
            detected = r.get('detected', {}).get(mod, False)
            colors_alpha.append(1.0 if detected else 0.3)

        color = MODALITY_COLORS.get(mod, '#666')
        bars = ax.bar(x + j * bar_width, dr2_vals, bar_width,
                      label=MOD_SHORT[mod], color=color)
        for bar, alpha in zip(bars, colors_alpha):
            bar.set_alpha(alpha)

    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(test_names, rotation=30, ha='right')
    ax.set_ylabel('Static dR2')
    ax.set_title('Detection: Static dR2 per Modality')
    ax.legend(fontsize=9)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Right: timing AUC-ROC values (coupled modalities only)
    ax = axes[1]
    auc_data = []
    auc_labels = []
    auc_colors = []
    for r in all_results:
        for mod in MODALITY_ORDER:
            auc_info = r.get('auc_results', {}).get(mod, {})
            auc = auc_info.get('auc')
            if auc is not None:
                auc_data.append(auc)
                auc_labels.append(f"{r['test']}\n{MOD_SHORT[mod]}")
                auc_colors.append(MODALITY_COLORS.get(mod, '#666'))

    if auc_data:
        bars = ax.barh(range(len(auc_data)), auc_data, color=auc_colors,
                       edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(auc_data)))
        ax.set_yticklabels(auc_labels, fontsize=9)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5,
                   label='Chance (0.5)')
        ax.axvline(0.8, color='green', linestyle='--', alpha=0.5,
                   label='Good (0.8)')
        ax.set_xlabel('AUC-ROC')
        ax.set_title('Timing Accuracy: AUC-ROC')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(output_dir, 'detection_and_timing_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_multi_seed_summary(seed_results, output_dir):
    """Box plot across seeds for each test x modality."""
    # Collect AUC values per (test, mod)
    from collections import defaultdict
    auc_by_test_mod = defaultdict(list)
    detect_by_test_mod = defaultdict(list)

    for seed, results in seed_results.items():
        for r in results:
            for mod in MODALITY_ORDER:
                key = (r['test'], mod)
                auc_info = r.get('auc_results', {}).get(mod, {})
                auc = auc_info.get('auc')
                if auc is not None:
                    auc_by_test_mod[key].append(auc)
                detected = r.get('detected', {}).get(mod, False)
                detect_by_test_mod[key].append(detected)

    if not auc_by_test_mod:
        return None

    # Build figure
    tests = list(dict.fromkeys(r['test'] for results in seed_results.values()
                               for r in results))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(tests) * 1.2)))

    # Left: AUC box plots
    ax = axes[0]
    positions = []
    labels_list = []
    data_list = []
    colors_list = []
    pos = 0
    for test in tests:
        for mod in MODALITY_ORDER:
            key = (test, mod)
            if key in auc_by_test_mod:
                data_list.append(auc_by_test_mod[key])
                labels_list.append(f'{test}\n{MOD_SHORT[mod]}')
                colors_list.append(MODALITY_COLORS.get(mod, '#666'))
                positions.append(pos)
                pos += 1
        pos += 0.5  # gap between tests

    if data_list:
        bp = ax.boxplot(data_list, positions=positions, widths=0.6,
                        patch_artist=True, vert=False)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels_list, fontsize=8)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.4)
        ax.axvline(0.8, color='green', linestyle='--', alpha=0.4)
        ax.set_xlabel('AUC-ROC')
        ax.set_title('Timing Accuracy Across Seeds')
        ax.set_xlim(0.3, 1.05)

    # Right: detection rate
    ax = axes[1]
    pos = 0
    bar_data = []
    bar_labels = []
    bar_colors = []
    for test in tests:
        for mod in MODALITY_ORDER:
            key = (test, mod)
            if key in detect_by_test_mod:
                rate = np.mean(detect_by_test_mod[key])
                bar_data.append(rate)
                bar_labels.append(f'{test}\n{MOD_SHORT[mod]}')
                bar_colors.append(MODALITY_COLORS.get(mod, '#666'))

    if bar_data:
        ax.barh(range(len(bar_data)), bar_data, color=bar_colors, alpha=0.7)
        ax.set_yticks(range(len(bar_data)))
        ax.set_yticklabels(bar_labels, fontsize=8)
        ax.set_xlabel('Detection Rate')
        ax.set_title('Detection Rate Across Seeds')
        ax.set_xlim(0, 1.1)

    fig.tight_layout()
    path = os.path.join(output_dir, 'multi_seed_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_single_test(estimator, test_name, kappa_dict, duty_override,
                    expected_coupled, duration, seed, output_dir):
    """Run one synthetic test, compute detection + timing accuracy."""
    coupled_mods = [m for m, k in kappa_dict.items() if k > 0]
    print(f"\n  Generating {duration}s session (seed={seed})...")

    t0 = time.time()
    session = build_synthetic_session_permod(
        duration, kappa_dict, seed=seed,
        duty_cycle_override=duty_override)
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    # Analyze
    t0 = time.time()
    result = estimator.analyze_session(session, 'p1_to_p2')
    analysis_time = time.time() - t0
    print(f"  Analysis: {analysis_time:.1f}s, "
          f"{result.n_significant_pathways} significant pathways")

    # Detection verdicts
    detected = {}
    dr2_values = {}
    for src_mod in MODALITY_ORDER:
        key = (src_mod, src_mod)
        detected[src_mod] = result.pathway_significant.get(key, False)
        if key in result.pathway_dr2:
            dr2_values[src_mod] = float(np.nanmean(result.pathway_dr2[key]))
        else:
            dr2_values[src_mod] = 0.0

    # Timing accuracy (AUC-ROC + correlation)
    auc_results = {}
    for mod in coupled_mods:
        auc, n_pos, n_neg, labels, scores = compute_timing_aucroc(
            result, session, mod)
        r_val, r_pval = compute_timing_correlation(result, session, mod)
        auc_results[mod] = {
            'auc': auc, 'n_pos': n_pos, 'n_neg': n_neg,
            'labels': labels, 'scores': scores,
            'pearson_r': float(r_val) if r_val is not None else None,
            'pearson_p': float(r_pval) if r_pval is not None else None,
        }

    # Print results
    print(f"\n  {'Modality':<10s} {'dR2':>10s} {'Detected':>10s} "
          f"{'AUC-ROC':>10s} {'Pearson r':>10s}")
    print(f"  {'-'*50}")
    for mod in MODALITY_ORDER:
        dr2_str = f"{dr2_values[mod]:+.6f}"
        det_str = 'YES' if detected[mod] else ''
        auc_info = auc_results.get(mod, {})
        auc_str = f"{auc_info['auc']:.3f}" if auc_info.get('auc') else '--'
        r_str = (f"{auc_info['pearson_r']:.3f}"
                 if auc_info.get('pearson_r') is not None else '--')
        print(f"  {MOD_SHORT[mod]:<10s} {dr2_str:>10s} {det_str:>10s} "
              f"{auc_str:>10s} {r_str:>10s}")

    # Pass/fail
    expected_pos = set(expected_coupled)
    expected_neg = set(MODALITY_ORDER) - expected_pos

    if test_name.endswith('_null'):
        false_pos = any(detected.get(m, False) for m in MODALITY_ORDER)
        passed = not false_pos
        status = 'PASS' if passed else 'FAIL (false positive)'
    else:
        true_pos = all(detected.get(m, False) for m in expected_pos)
        false_pos = any(detected.get(m, False) for m in expected_neg)
        if not true_pos and false_pos:
            status = 'FAIL (missed + false positive)'
            passed = False
        elif not true_pos:
            status = 'FAIL (missed coupling)'
            passed = False
        elif false_pos:
            status = 'FAIL (false positive)'
            passed = False
        else:
            status = 'PASS'
            passed = True

    print(f"  Result: {status}")

    # Visualizations (for first seed only — controlled by caller)
    if output_dir:
        plot_gate_overlay(result, session, output_dir, test_name)
        if auc_results:
            plot_roc_curves(auc_results, output_dir, test_name)

    return {
        'test': test_name,
        'seed': seed,
        'kappa_dict': kappa_dict,
        'duty_override': duty_override,
        'expected_coupled': expected_coupled,
        'detected': {k: bool(v) for k, v in detected.items()},
        'dr2_values': dr2_values,
        'auc_results': {k: {kk: vv for kk, vv in v.items()
                            if kk not in ('labels', 'scores')}
                        for k, v in auc_results.items()},
        'passed': passed,
        'status': status,
        'analysis_time': analysis_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description='CADENCE synthetic timing accuracy validation')
    parser.add_argument('--config', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--duration', type=int, default=3000,
                        help='Session duration in seconds (default: 3000)')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds per test (default: 3)')
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--output', default='results/cadence_synthetic_timing')
    args = parser.parse_args()

    if args.config is None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    # Disable Phase 2 decomposition — timing test only needs Phase 1
    # (same-modality dR2 for AUC-ROC). Saves ~15s per significant pathway.
    config['decomposition']['enabled'] = False

    os.makedirs(args.output, exist_ok=True)
    estimator = CouplingEstimator(config)

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    seed_results = {}
    all_flat = []

    total_t0 = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"# Seed {seed} ({seed_idx+1}/{args.n_seeds})")
        print(f"{'#'*60}")
        seed_results[seed] = []

        for test_name, kappa_dict, duty_override, expected_coupled in TIMING_TESTS:
            print(f"\n{'='*60}")
            print(f"Test {test_name} (seed={seed})")
            coupled_mods = [m for m, k in kappa_dict.items() if k > 0]
            print(f"  Coupled: {[MOD_SHORT[m] for m in coupled_mods] if coupled_mods else 'NONE (null)'}")
            print(f"{'='*60}")

            # Only generate detailed plots for first seed
            plot_dir = args.output if seed_idx == 0 else None

            r = run_single_test(
                estimator, test_name, kappa_dict, duty_override,
                expected_coupled, args.duration, seed, plot_dir)

            seed_results[seed].append(r)
            all_flat.append(r)

    total_time = time.time() - total_t0

    # -----------------------------------------------------------------------
    # Aggregate across seeds
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print(f"AGGREGATE RESULTS ({args.n_seeds} seeds x {len(TIMING_TESTS)} tests, "
          f"{args.duration}s sessions)")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")

    # Per-test aggregate
    for test_name, _, _, expected_coupled in TIMING_TESTS:
        test_runs = [r for r in all_flat if r['test'] == test_name]
        n_pass = sum(1 for r in test_runs if r['passed'])
        print(f"\n  {test_name}: {n_pass}/{len(test_runs)} seeds pass")

        # Detection rates
        for mod in MODALITY_ORDER:
            detect_rate = np.mean([r['detected'][mod] for r in test_runs])
            mean_dr2 = np.mean([r['dr2_values'][mod] for r in test_runs])
            expected = mod in expected_coupled
            verdict = 'TP' if (expected and detect_rate > 0.5) else \
                      'FN' if (expected and detect_rate <= 0.5) else \
                      'FP' if (not expected and detect_rate > 0.5) else 'TN'
            print(f"    {MOD_SHORT[mod]:<6s}: detect={detect_rate:.0%}, "
                  f"dR2={mean_dr2:+.6f}  [{verdict}]", end='')

            # AUC if coupled
            aucs = [r['auc_results'].get(mod, {}).get('auc')
                    for r in test_runs]
            aucs = [a for a in aucs if a is not None]
            if aucs:
                print(f"  AUC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}", end='')

            corrs = [r['auc_results'].get(mod, {}).get('pearson_r')
                     for r in test_runs]
            corrs = [c for c in corrs if c is not None]
            if corrs:
                print(f"  r={np.mean(corrs):.3f}", end='')
            print()

    # Overall pass rate
    n_total = len(all_flat)
    n_pass = sum(1 for r in all_flat if r['passed'])
    print(f"\n  Overall: {n_pass}/{n_total} test-seeds pass ({n_pass/n_total:.0%})")

    # -----------------------------------------------------------------------
    # Generate summary plots
    # -----------------------------------------------------------------------
    print("\nGenerating summary visualizations...")

    # Use first seed results for per-test plots
    first_seed_results = seed_results[seeds[0]]
    plot_summary_heatmap(first_seed_results, args.output)
    plot_detection_summary(first_seed_results, args.output)
    if args.n_seeds > 1:
        plot_multi_seed_summary(seed_results, args.output)

    # Save full JSON results
    # Strip labels/scores from auc_results to keep JSON manageable
    json_results = {
        'config': {
            'duration': args.duration,
            'n_seeds': args.n_seeds,
            'base_seed': args.base_seed,
            'n_surrogates': config['significance']['surrogate'].get(
                'n_screen_surrogates', 200),
        },
        'seeds': {str(s): rs for s, rs in seed_results.items()},
        'aggregate': {
            'n_pass': n_pass,
            'n_total': n_total,
            'pass_rate': n_pass / n_total,
            'total_time_s': total_time,
        },
    }
    json_path = os.path.join(args.output, 'timing_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nResults saved: {json_path}")
    print(f"Plots saved to: {args.output}/")


if __name__ == '__main__':
    main()
