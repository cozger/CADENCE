"""CADENCE semi-synthetic validation with AUC.

Real features from cross-dyad pairs (P1 from dyad A, P2 from dyad B)
guarantee zero baseline coupling while preserving real rank, autocorrelation,
and spectral structure. Coupling is injected at known kappa levels and the
pipeline's detection is evaluated via ROC/AUC.

Performance:
  - Base session (with interbrain CWT) built ONCE per pair, reused for all
    kappa/modality combos.  ~20x fewer GPU CWT calls vs naive approach.
  - inject_coupling_modality() is CPU-only numpy — microseconds per call.
  - --pair-indices flag enables GPU-level parallelism in the sbatch script.
  - Incremental JSON saves after each analysis survive job preemption.

Usage:
    python scripts/run_semisynthetic.py
    python scripts/run_semisynthetic.py --n-pairs 4 --device cuda:0
    python scripts/run_semisynthetic.py --pair-indices 0,1 --device cuda:0
    python scripts/run_semisynthetic.py --kappas 0.0,0.4,0.6,0.8
"""

import argparse
import gc
import json
import os
import sys
import time

os.environ['PYTHONUNBUFFERED'] = '1'

print("=== CADENCE V3 Semisynthetic (GPD + HMM + stability_hmm fallback) ===",
      flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.coupling.estimator import CouplingEstimator
from cadence.significance.detection import detection_summary
from cadence.synthetic import (
    find_valid_window, build_semisynthetic_base,
    inject_coupling_modality, inject_coupling_all,
)
from cadence.constants import MODALITY_ORDER_V2, MOD_SHORT_V2, MODALITY_SPECS_V2


# Target modality configs: each injects coupling into ONE modality
TARGET_MODS = [x for x in ['eeg_wavelet', 'blendshapes_v2', 'pose_features']
               if x in (os.environ.get('CADENCE_TARGET_MODS', 'eeg_wavelet,blendshapes_v2,pose_features').split(','))]

# Same-modality pathway keys for AUC
SAME_MOD_PATHWAYS = {mod: (mod, mod) for mod in TARGET_MODS}


def compute_timing_metrics(coupling_gate, coupling_posterior, gate_hz,
                           eval_times, threshold=0.5):
    """Compare HMM coupling posterior against known injection gate.

    Args:
        coupling_gate: (N_gate,) known gate in [0,1] at gate_hz.
        coupling_posterior: (T_eval,) HMM posterior P(coupled) at eval_times.
        gate_hz: Sampling rate of the coupling gate.
        eval_times: (T_eval,) timestamps of the posterior.
        threshold: Binarization threshold for both gate and posterior.

    Returns:
        dict with timing metrics.
    """
    if coupling_gate is None or coupling_posterior is None:
        return {}
    if len(coupling_posterior) < 10:
        return {}

    # Resample gate to eval_times
    gate_times = np.arange(len(coupling_gate)) / gate_hz
    gate_resampled = np.interp(eval_times, gate_times, coupling_gate)

    # Binarize
    gate_on = gate_resampled > threshold
    post_on = coupling_posterior > threshold

    n = len(eval_times)
    if n == 0:
        return {}

    # Hit rate: fraction of gate-on timepoints where posterior is also on
    n_gate_on = np.sum(gate_on)
    n_post_on = np.sum(post_on)

    if n_gate_on > 0:
        hit_rate = float(np.sum(gate_on & post_on) / n_gate_on)
    else:
        hit_rate = float('nan')

    # False alarm rate: fraction of gate-off timepoints where posterior is on
    n_gate_off = n - n_gate_on
    if n_gate_off > 0:
        false_alarm = float(np.sum(~gate_on & post_on) / n_gate_off)
    else:
        false_alarm = float('nan')

    # IoU (intersection over union)
    intersection = np.sum(gate_on & post_on)
    union = np.sum(gate_on | post_on)
    iou = float(intersection / union) if union > 0 else 0.0

    # Temporal correlation (Pearson r between continuous signals)
    if np.std(gate_resampled) > 1e-10 and np.std(coupling_posterior) > 1e-10:
        corr = float(np.corrcoef(gate_resampled, coupling_posterior)[0, 1])
    else:
        corr = 0.0

    return {
        'hit_rate': hit_rate,
        'false_alarm': false_alarm,
        'iou': iou,
        'temporal_corr': corr,
        'n_gate_on': int(n_gate_on),
        'n_post_on': int(n_post_on),
        'gate_duty': float(n_gate_on / n) if n > 0 else 0.0,
        'post_duty': float(n_post_on / n) if n > 0 else 0.0,
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def build_cross_dyad_pairs(sessions, n_pairs, seed=42):
    """Build cross-dyad pairs from available sessions.

    Returns list of (name_a, session_a, window_a, name_b, session_b, window_b).
    """
    rng = np.random.default_rng(seed)

    # Filter to sessions with valid 30-min windows
    valid_sessions = []
    for name, cache_path, session in sessions:
        print(f"  {name}:", flush=True)
        window = find_valid_window(session, min_duration=1800,
                                    verbose=True)
        if window is not None:
            valid_sessions.append((name, session, window))
            print(f"    -> valid window "
                  f"[{window[0]:.0f}s, {window[1]:.0f}s]", flush=True)
        else:
            print(f"    -> SKIPPED", flush=True)

    if len(valid_sessions) < 2:
        print("ERROR: Need at least 2 sessions with valid windows", flush=True)
        return []

    # Build pairs (A != B)
    indices = list(range(len(valid_sessions)))
    pairs = []
    rng.shuffle(indices)

    for i in range(0, len(indices) - 1, 2):
        if len(pairs) >= n_pairs:
            break
        a_idx, b_idx = indices[i], indices[i + 1]
        name_a, sess_a, win_a = valid_sessions[a_idx]
        name_b, sess_b, win_b = valid_sessions[b_idx]
        pairs.append((name_a, sess_a, win_a, name_b, sess_b, win_b))

    # Wrap around if we need more pairs
    if len(pairs) < n_pairs and len(valid_sessions) >= 2:
        for k in range(n_pairs - len(pairs)):
            a_idx = k % len(valid_sessions)
            b_idx = (k + 1) % len(valid_sessions)
            if a_idx == b_idx:
                b_idx = (b_idx + 1) % len(valid_sessions)
            name_a, sess_a, win_a = valid_sessions[a_idx]
            name_b, sess_b, win_b = valid_sessions[b_idx]
            pairs.append((name_a, sess_a, win_a, name_b, sess_b, win_b))

    return pairs[:n_pairs]


def compute_auc(labels, scores):
    """Compute AUC from labels (0/1) and scores (p-values).

    Lower p-value = more likely coupled, so we negate before AUC.
    """
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return float('nan')
        return roc_auc_score(labels, [-s for s in scores])
    except ImportError:
        return _manual_auc(labels, scores)


def _manual_auc(labels, scores):
    """Manual AUC without sklearn."""
    if len(set(labels)) < 2:
        return float('nan')
    paired = sorted(zip([-s for s in scores], labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    tp = 0
    auc = 0.0
    for _score, label in paired:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def plot_roc_curves(roc_data, output_path):
    """Plot ROC curves per modality and save."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping ROC plot", flush=True)
        return

    fig, axes = plt.subplots(
        1, len(TARGET_MODS), figsize=(5 * len(TARGET_MODS), 5))
    if len(TARGET_MODS) == 1:
        axes = [axes]

    colors = {'eeg_wavelet': '#2196F3', 'blendshapes_v2': '#4CAF50',
              'pose_features': '#FF9800'}

    for ax, mod in zip(axes, TARGET_MODS):
        if mod not in roc_data:
            continue
        labels = roc_data[mod]['labels']
        scores = roc_data[mod]['scores']
        auc_val = roc_data[mod]['auc']
        short = MOD_SHORT_V2.get(mod, mod)
        color = colors.get(mod, '#333333')

        if len(set(labels)) >= 2:
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(labels, [-s for s in scores])
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f'{short} (AUC={auc_val:.2f})')
            except ImportError:
                ax.text(0.5, 0.5, f'AUC={auc_val:.2f}',
                        ha='center', va='center', transform=ax.transAxes)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'{short} (AUC={auc_val:.2f})')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC plot saved: {output_path}", flush=True)


def _save_incremental(output_dir, all_results, roc_data, pairs_info):
    """Save current results to JSON (survives crashes)."""
    output = {
        'n_analyses': len(all_results),
        'analyses': all_results,
        'roc_data': {mod: {'labels': d['labels'], 'scores': d['scores']}
                     for mod, d in roc_data.items()},
        'pairs': pairs_info,
    }
    path = os.path.join(output_dir, 'semisynthetic_results.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(
        description='CADENCE semi-synthetic validation with AUC')
    parser.add_argument('--config', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--n-pairs', type=int, default=4,
                        help='Total number of cross-dyad pairs to build')
    parser.add_argument('--pair-indices', default=None,
                        help='Comma-separated pair indices to run '
                             '(e.g. 0,1 for first two). '
                             'Enables GPU-level parallelism in sbatch.')
    parser.add_argument('--kappas', default='0.0,0.2,0.4,0.6,0.8',
                        help='Comma-separated kappa levels')
    parser.add_argument('--output', default='results/semisynthetic',
                        help='Output directory')
    parser.add_argument('--cache-dir', default=None,
                        help='Session cache directory')
    parser.add_argument('--pair-seed', type=int, default=42,
                        help='Seed for cross-dyad pairing')
    parser.add_argument('--no-combined', action='store_true',
                        help='Skip combined (all-modality) tests')
    parser.add_argument('--duty-cycle', type=float, default=None,
                        help='Override coupling duty cycle (fraction of time coupled)')
    parser.add_argument('--target-mods', default=None,
                        help='Comma-separated target modalities to test '
                             '(default: all enabled in CADENCE_TARGET_MODS)')
    args = parser.parse_args()

    # Load config
    if args.config is None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'default.yaml')
        if os.path.exists(default_path):
            args.config = default_path

    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    # Disable ECG moderation for semi-synthetic (cross-dyad ECG is unrelated)
    config['stage2']['moderation']['enabled'] = False

    kappa_levels = [float(k) for k in args.kappas.split(',')]

    # Override target modalities if specified
    global TARGET_MODS, SAME_MOD_PATHWAYS
    if args.target_mods:
        TARGET_MODS = [m.strip() for m in args.target_mods.split(',')]
        SAME_MOD_PATHWAYS = {mod: (mod, mod) for mod in TARGET_MODS}

    duty_cycle_override = args.duty_cycle

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = args.cache_dir or config.get('session_cache',
                                              'session_cache')

    print(f"Semi-synthetic validation", flush=True)
    print(f"  Pairs: {args.n_pairs}", flush=True)
    print(f"  Kappas: {kappa_levels}", flush=True)
    print(f"  Target mods: {TARGET_MODS}", flush=True)
    if duty_cycle_override is not None:
        print(f"  Duty cycle override: {duty_cycle_override:.1%}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Cache: {cache_dir}", flush=True)

    # Discover and load sessions
    print("\nDiscovering sessions...", flush=True)
    cached = discover_cached_sessions(cache_dir)
    excluded = set(config.get('excluded_sessions', []))
    cached = [(n, p) for n, p in cached if n not in excluded]
    print(f"  Found {len(cached)} cached sessions", flush=True)

    print("\nLoading sessions (with v2 features)...", flush=True)
    sessions = []
    for name, cache_path in cached:
        try:
            session = load_session_from_cache(cache_path, config=config)
            sessions.append((name, cache_path, session))
            print(f"  Loaded {name} "
                  f"({session.get('duration', 0):.0f}s)", flush=True)
        except Exception as e:
            print(f"  Failed to load {name}: {e}", flush=True)

    # Build cross-dyad pairs (all pairs, regardless of --pair-indices)
    print(f"\nBuilding {args.n_pairs} cross-dyad pairs...", flush=True)
    all_pairs = build_cross_dyad_pairs(
        sessions, args.n_pairs, seed=args.pair_seed)
    if not all_pairs:
        print("ERROR: No valid pairs found", flush=True)
        sys.exit(1)
    print(f"  Built {len(all_pairs)} pairs", flush=True)

    # Filter to requested pair indices
    if args.pair_indices is not None:
        indices = [int(i) for i in args.pair_indices.split(',')]
        pairs = [all_pairs[i] for i in indices if i < len(all_pairs)]
        print(f"  Running pair indices {indices} "
              f"({len(pairs)} pairs)", flush=True)
    else:
        pairs = all_pairs

    # Free sessions we don't need (reduce memory)
    needed_sessions = set()
    for name_a, _, _, name_b, _, _ in pairs:
        needed_sessions.add(name_a)
        needed_sessions.add(name_b)
    del sessions  # free all; pairs hold references to the data

    # Pre-count analyses
    n_per_mod = len(TARGET_MODS)
    n_combined = 0 if args.no_combined else 1
    analyses_per_pair = len(kappa_levels) * (n_per_mod + n_combined)
    total_analyses = len(pairs) * analyses_per_pair
    print(f"\nTotal analyses: {total_analyses} "
          f"({len(pairs)} pairs x {len(kappa_levels)} kappas x "
          f"({n_per_mod} mods + {n_combined} combined))", flush=True)

    # Run sweep
    estimator = CouplingEstimator(config)

    roc_data = {mod: {'labels': [], 'scores': []} for mod in TARGET_MODS}
    all_results = []
    pairs_info = []

    analysis_idx = 0
    t_start_all = time.time()

    for pair_idx_local, (name_a, sess_a, win_a,
                         name_b, sess_b, win_b) in enumerate(pairs):
        t_start_pair = win_a[0]
        t_end_pair = win_a[1]

        # Global pair index (for reproducible seeds across GPU splits)
        if args.pair_indices is not None:
            global_pair_idx = int(args.pair_indices.split(',')[pair_idx_local])
        else:
            global_pair_idx = pair_idx_local

        pairs_info.append({
            'pair_idx': global_pair_idx,
            'p1_session': name_a, 'p2_session': name_b,
            'window': [t_start_pair, t_end_pair],
        })

        print(f"\n{'='*60}", flush=True)
        print(f"Pair {global_pair_idx}: P1={name_a}, P2={name_b}", flush=True)
        print(f"  Window: [{t_start_pair:.0f}s, {t_end_pair:.0f}s]", flush=True)

        # Build base session ONCE per pair (interbrain CWT computed here)
        t0_base = time.time()
        base_session = build_semisynthetic_base(
            sess_a, sess_b, t_start_pair, t_end_pair)
        dt_base = time.time() - t0_base
        has_ib = 'eeg_interbrain' in base_session
        print(f"  Base session built in {dt_base:.1f}s "
              f"(interbrain={'yes' if has_ib else 'no'})", flush=True)
        print(f"{'='*60}", flush=True)

        for kappa in kappa_levels:
            # Per-modality tests
            for target_mod in TARGET_MODS:
                analysis_idx += 1
                short = MOD_SHORT_V2.get(target_mod, target_mod)
                label = 1 if kappa > 0 else 0

                print(f"\n  [{analysis_idx}/{total_analyses}] "
                      f"kappa={kappa:.1f} target={short}", flush=True)

                # Fast: inject coupling into one modality (CPU-only)
                seed = (args.pair_seed + global_pair_idx * 1000
                        + int(kappa * 100))
                semi_session = inject_coupling_modality(
                    base_session, target_mod, kappa,
                    lag_s=2.0, seed=seed,
                    duty_cycle=duty_cycle_override)

                t0 = time.time()
                try:
                    result = estimator.analyze_session(
                        semi_session, 'p1_to_p2')
                except Exception as _run_err:
                    print(f"    ANALYSIS FAILED: {_run_err}", flush=True)
                    result = None
                dt = time.time() - t0

                # Free GPU memory between runs to prevent cumulative VRAM pressure
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                if result is None:
                    all_results.append({
                        'pair_idx': global_pair_idx,
                        'pair': f'{name_a}_x_{name_b}',
                        'kappa': kappa,
                        'target_mod': target_mod,
                        'label': label,
                        'p_value': 1.0, 'detected': False,
                        'mean_dr2': 0.0, 'best_pathway': 'error',
                        'n_targeting': 0, 'n_detected': 0,
                        'timing': {}, 'analysis_time_s': dt,
                    })
                    _save_incremental(output_dir, all_results,
                                      roc_data, pairs_info)
                    continue

                summary = detection_summary(result, config)

                # Score ALL pathways targeting the injected modality
                # (not just same-source). Any src→target_mod pathway
                # can detect the injected coupling.
                # Get coupling gate for timing analysis
                coupling_gate = semi_session.get(
                    'coupling_gates', {}).get(target_mod)
                gate_hz = MODALITY_SPECS_V2[target_mod][1] if \
                    target_mod in MODALITY_SPECS_V2 else 10.0

                # Score ALL pathways targeting the injected modality
                best_p = 1.0
                best_det = False
                best_dr2 = 0.0
                best_key = None
                n_targeting = 0
                n_detected = 0
                pw_timing = {}
                for key, det_info in summary.items():
                    s_mod, t_mod = key
                    if t_mod != target_mod:
                        continue
                    # Skip interbrain→eeg_wavelet (structural confound:
                    # PLV contains target EEG by construction)
                    if s_mod == 'eeg_interbrain' and 'eeg' in t_mod:
                        continue
                    n_targeting += 1
                    p = det_info.get('p_value', 1.0)
                    d = det_info.get('detected', False)
                    dr2 = det_info.get('mean_dr2', 0.0)
                    if d:
                        n_detected += 1
                    s_short = MOD_SHORT_V2.get(s_mod, s_mod)
                    t_short = MOD_SHORT_V2.get(t_mod, t_mod)
                    status = 'DETECTED' if d else ''

                    # Timing accuracy
                    timing = {}
                    posterior = getattr(result, 'pathway_coupling_posterior',
                                        {}).get(key)
                    pw_times = getattr(result, 'pathway_times',
                                        {}).get(key, result.times)
                    if posterior is not None and coupling_gate is not None:
                        timing = compute_timing_metrics(
                            coupling_gate, posterior, gate_hz, pw_times)
                        pw_timing[f"{s_short}->{t_short}"] = timing
                        timing_str = (f" hit={timing.get('hit_rate',0):.0%}"
                                      f" fa={timing.get('false_alarm',0):.0%}"
                                      f" IoU={timing.get('iou',0):.2f}"
                                      f" r={timing.get('temporal_corr',0):.2f}")
                    else:
                        timing_str = ""

                    print(f"    {s_short}->{t_short}: p={p:.4f} "
                          f"dR2={dr2:.6f} {status}{timing_str}",
                          flush=True)

                    if p < best_p:
                        best_p = p
                        best_det = d
                        best_dr2 = dr2
                        best_key = key

                # Fallback to same-mod if nothing found
                if best_key is None:
                    same_key = SAME_MOD_PATHWAYS[target_mod]
                    if same_key in summary:
                        det_info = summary[same_key]
                        best_p = det_info.get('p_value', 1.0)
                        best_det = det_info.get('detected', False)
                        best_dr2 = det_info.get('mean_dr2', 0.0)

                p_val = best_p
                detected = best_det
                mean_dr2 = best_dr2

                roc_data[target_mod]['labels'].append(label)
                roc_data[target_mod]['scores'].append(p_val)

                # Summary line
                hit_frac = (f"{n_detected}/{n_targeting}"
                            if n_targeting > 0 else "0/0")
                if best_key:
                    bk_short = (f"{MOD_SHORT_V2.get(best_key[0], best_key[0])}"
                                f"->{MOD_SHORT_V2.get(best_key[1], best_key[1])}")
                    print(f"    => Hit {hit_frac} pathways | "
                          f"best={bk_short} p={p_val:.4f} ({dt:.1f}s)",
                          flush=True)
                else:
                    print(f"    => No pathways targeting {short} ({dt:.1f}s)",
                          flush=True)

                all_results.append({
                    'pair_idx': global_pair_idx,
                    'pair': f'{name_a}_x_{name_b}',
                    'kappa': kappa,
                    'target_mod': target_mod,
                    'label': label,
                    'p_value': p_val,
                    'detected': detected,
                    'mean_dr2': mean_dr2,
                    'best_pathway': (f"{best_key[0]}->{best_key[1]}"
                                     if best_key else 'none'),
                    'n_targeting': n_targeting,
                    'n_detected': n_detected,
                    'timing': pw_timing,
                    'analysis_time_s': dt,
                })

                # Incremental save
                _save_incremental(output_dir, all_results,
                                  roc_data, pairs_info)

            # Combined test
            if not args.no_combined:
                analysis_idx += 1
                print(f"\n  [{analysis_idx}/{total_analyses}] "
                      f"kappa={kappa:.1f} target=ALL", flush=True)

                seed = (args.pair_seed + global_pair_idx * 1000 + 999)
                semi_session = inject_coupling_all(
                    base_session, kappa, lag_s=2.0, seed=seed)

                t0 = time.time()
                try:
                    result = estimator.analyze_session(
                        semi_session, 'p1_to_p2')
                except Exception as _run_err:
                    print(f"    COMBINED ANALYSIS FAILED: {_run_err}",
                          flush=True)
                    result = None
                dt = time.time() - t0

                # Free GPU memory between runs
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                if result is None:
                    all_results.append({
                        'pair_idx': global_pair_idx,
                        'pair': f'{name_a}_x_{name_b}',
                        'kappa': kappa,
                        'target_mod': 'combined',
                        'label': 1 if kappa > 0 else 0,
                        'analysis_time_s': dt,
                    })
                    _save_incremental(output_dir, all_results,
                                      roc_data, pairs_info)
                    continue

                summary = detection_summary(result, config)
                for mod in TARGET_MODS:
                    pw_key = SAME_MOD_PATHWAYS[mod]
                    short_m = MOD_SHORT_V2.get(mod, mod)
                    if pw_key in summary:
                        det = summary[pw_key]
                        print(f"    {short_m}->{short_m}: "
                              f"p={det.get('p_value',1):.4f} "
                              f"dR2={det.get('mean_dr2',0):.6f} "
                              f"{'DETECTED' if det.get('detected') else ''}",
                              flush=True)

                all_results.append({
                    'pair_idx': global_pair_idx,
                    'pair': f'{name_a}_x_{name_b}',
                    'kappa': kappa,
                    'target_mod': 'combined',
                    'label': 1 if kappa > 0 else 0,
                    'analysis_time_s': dt,
                })

                _save_incremental(output_dir, all_results,
                                  roc_data, pairs_info)

        # Free base session to reduce memory between pairs
        del base_session

    # ── Final results ─────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("AUC RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    auc_results = {}
    for mod in TARGET_MODS:
        labels = roc_data[mod]['labels']
        scores = roc_data[mod]['scores']
        auc_val = compute_auc(labels, scores)
        roc_data[mod]['auc'] = auc_val
        auc_results[mod] = auc_val
        short = MOD_SHORT_V2.get(mod, mod)
        print(f"  {short:>6s}: AUC = {auc_val:.3f} "
              f"(n_pos={sum(labels)}, "
              f"n_neg={len(labels)-sum(labels)})", flush=True)

    # FPR at kappa=0
    print(f"\nFPR at kappa=0:", flush=True)
    for mod in TARGET_MODS:
        null_results = [r for r in all_results
                        if r.get('target_mod') == mod
                        and r['kappa'] == 0.0]
        n_fp = sum(1 for r in null_results if r.get('detected', False))
        n_total = len(null_results)
        fpr = n_fp / max(n_total, 1)
        short = MOD_SHORT_V2.get(mod, mod)
        print(f"  {short:>6s}: {n_fp}/{n_total} = {fpr:.1%}", flush=True)

    # TPR at kappa=0.6
    if 0.6 in kappa_levels:
        print(f"\nTPR at kappa=0.6:", flush=True)
        for mod in TARGET_MODS:
            coupled = [r for r in all_results
                       if r.get('target_mod') == mod
                       and r['kappa'] == 0.6]
            n_tp = sum(1 for r in coupled if r.get('detected', False))
            n_total = len(coupled)
            tpr = n_tp / max(n_total, 1)
            short = MOD_SHORT_V2.get(mod, mod)
            print(f"  {short:>6s}: {n_tp}/{n_total} = {tpr:.1%}",
                  flush=True)

    total_time = time.time() - t_start_all
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)",
          flush=True)

    # Save final results
    output = {
        'n_pairs': len(pairs),
        'kappa_levels': kappa_levels,
        'auc': auc_results,
        'analyses': all_results,
        'roc_data': {mod: {'labels': d['labels'], 'scores': d['scores'],
                           'auc': d['auc']}
                     for mod, d in roc_data.items()},
        'pairs': pairs_info,
    }

    json_path = os.path.join(output_dir, 'semisynthetic_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved: {json_path}", flush=True)

    # ROC plot
    plot_roc_curves(roc_data, os.path.join(output_dir, 'roc_curves.png'))

    # Verification
    print(f"\n{'='*60}", flush=True)
    print("VERIFICATION", flush=True)
    print(f"{'='*60}", flush=True)
    all_pass = True
    for mod in TARGET_MODS:
        short = MOD_SHORT_V2.get(mod, mod)
        auc = auc_results.get(mod, 0)
        if auc < 0.80:
            print(f"  [WARN] {short} AUC={auc:.3f} < 0.80", flush=True)
            all_pass = False
        else:
            print(f"  [ OK ] {short} AUC={auc:.3f} >= 0.80", flush=True)

    null_results = [r for r in all_results if r['kappa'] == 0.0
                    and r.get('target_mod') != 'combined']
    n_fp = sum(1 for r in null_results if r.get('detected', False))
    n_null = len(null_results)
    fpr = n_fp / max(n_null, 1)
    if fpr > 0.10:
        print(f"  [WARN] Overall null FPR={fpr:.1%} > 10%", flush=True)
        all_pass = False
    else:
        print(f"  [ OK ] Overall null FPR={fpr:.1%} <= 10%", flush=True)

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\n{status}", flush=True)


if __name__ == '__main__':
    main()
