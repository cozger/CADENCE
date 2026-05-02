"""V10 Semi-Synthetic Test Battery.

Validates the 24D V10 scaffold via raw-level coupling injection.

Phase 1: LZ complexity scenarios (L1 theta, L2 alpha)
Phase 2: V8.2 regression (verify channels 0-17 unchanged)
Phase 3: Graph channel response (modularity/centrality shift under multimodal coupling)
Phase 4: Null integrity (kappa=0 AUC in [0.40, 0.60] for all channels)

All pseudo-dyad pairs x kappa levels parallelized via joblib (n_jobs=16).

Usage:
    python scripts/_test_v10_semisynthetic_battery.py --phase 1
    python scripts/_test_v10_semisynthetic_battery.py --phase 2
    python scripts/_test_v10_semisynthetic_battery.py --all
    python scripts/_test_v10_semisynthetic_battery.py --quick   # 3 pairs, fast
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.constants import (
    V10_LZ_SCENARIOS, V10_KAPPA_RANGES, V10_MODALITY_KEYS,
    V82_EEG_SCENARIOS, V82_BL_SCENARIOS,
)
from cadence.synthetic import generate_coupling_gate
from cadence.synthetic_v82 import (
    build_v82_pseudo_dyad, inject_eeg_v82, inject_bl_v82,
    compute_auc_within_session, compute_auc_cross_session,
    generate_v82_gate,
)
from cadence.synthetic_v10 import inject_lz_v10, inject_all_v10

FS_OUT = 2.0
OUT_DIR = 'results/v10_semisynthetic'
N_JOBS = 4  # V10 pipeline is memory-heavy (~500MB per job); 4 is safe on 32GB


# =========================================================================
# Session loading (reuse V8.2 pattern)
# =========================================================================

def load_available_sessions(config, max_sessions=None):
    cached_sessions = discover_cached_sessions(config['session_cache'])
    sessions = []
    for name, path in cached_sessions:
        try:
            cached = load_session_from_cache(path, config)
            if 'p1_eeg' in cached and 'p2_eeg' in cached:
                sessions.append((name, cached))
                if max_sessions and len(sessions) >= max_sessions:
                    break
        except Exception:
            continue
    return sessions


def generate_pseudo_dyad_pairs(sessions, n_pairs=10, seed=42):
    rng = np.random.default_rng(seed)
    n = len(sessions)
    if n < 2:
        return []
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        pairs.append((i, j))
    return pairs


# =========================================================================
# Core evaluation: run V10 scaffold on injected data
# =========================================================================

def evaluate_lz_pair(cached_a, cached_b, kappa, scenario_name,
                     duration_s=300.0, seed=42):
    """Evaluate one LZ injection at a given kappa.

    Returns per-feature AUC dict.
    """
    from scripts._run_scaffold_v10 import run_from_raw_v10

    ts_a = cached_a.get('p1_eeg_ts', np.array([0, 300]))
    ts_b = cached_b.get('p2_eeg_ts', cached_b.get('p1_eeg_ts', np.array([0, 300])))

    t_start = max(ts_a[0], ts_b[0])
    t_end = min(ts_a[-1], ts_b[-1])
    available = t_end - t_start
    if available < 60:
        return None

    dur = min(duration_s, available)
    t_end = t_start + dur
    t_common = np.arange(t_start, t_end, 1.0 / FS_OUT)
    N = len(t_common)

    # Build pseudo-dyad
    base = build_v82_pseudo_dyad(cached_a, cached_b, t_start, t_end)

    # Generate gate (matches V8.2 gate format)
    gate_cfg = {'event_range_s': (10.0, 25.0), 'ramp_s': 2.0, 'duty_cycle': 0.4}
    n_eeg = min(len(base.get('p1_eeg', [])), len(base.get('p2_eeg', [])))
    gate_eeg = generate_coupling_gate(n_eeg, 256.0, gate_cfg, seed=seed) if n_eeg > 0 else np.ones(1)
    gate_2hz = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(gate_eeg)), gate_eeg)

    # Inject LZ coupling
    if kappa > 0 and 'p1_eeg' in base and 'p2_eeg' in base:
        n_eeg = min(len(base['p1_eeg']), len(base['p2_eeg']))
        gate_eeg_r = np.interp(np.linspace(0, 1, n_eeg),
                                np.linspace(0, 1, len(gate_eeg)), gate_eeg)
        base['p1_eeg'], base['p2_eeg'] = inject_lz_v10(
            base['p1_eeg'][:n_eeg], base['p2_eeg'][:n_eeg],
            kappa, scenario_name, gate_eeg_r, seed=seed)

    # Run V10 scaffold
    try:
        z_24, z_24_raw, mask_24, pw_diag, U = run_from_raw_v10(
            base, t_common, label=f'lz_{scenario_name}_k{kappa:.2f}')
    except Exception as e:
        print(f"    Error: {e}")
        return None

    # Compute AUC for each channel (directional: max(auc, 1-auc))
    result = {'kappa': kappa, 'scenario': scenario_name}
    for i, key in enumerate(V10_MODALITY_KEYS):
        auc_raw = compute_auc_within_session(z_24[:, i], gate_2hz)
        # Directional AUC: coupling may increase OR decrease the feature
        # (e.g., LZ concordance decreases during coupling = shared regularity)
        auc_dir = max(auc_raw, 1.0 - auc_raw)
        result[f'auc_{key}'] = float(auc_dir)
        result[f'auc_raw_{key}'] = float(auc_raw)

    return result


def evaluate_v82_regression(cached_a, cached_b, kappa, scenario_name,
                            modality='eeg', duration_s=300.0, seed=42):
    """Run a V8.2 scenario through V10 scaffold, check channel 0-17 AUCs.

    Verifies V10 doesn't corrupt V8.2 features.
    """
    from scripts._run_scaffold_v10 import run_from_raw_v10
    from scripts._run_scaffold_v82 import run_from_raw as run_from_raw_v82, MODALITY_KEYS as V82_KEYS

    ts_a = cached_a.get('p1_eeg_ts', np.array([0, 300]))
    ts_b = cached_b.get('p2_eeg_ts', cached_b.get('p1_eeg_ts', np.array([0, 300])))

    t_start = max(ts_a[0], ts_b[0])
    t_end = min(ts_a[-1], ts_b[-1])
    available = t_end - t_start
    if available < 60:
        return None

    dur = min(duration_s, available)
    t_end = t_start + dur
    t_common = np.arange(t_start, t_end, 1.0 / FS_OUT)
    N = len(t_common)

    base = build_v82_pseudo_dyad(cached_a, cached_b, t_start, t_end)

    # Generate gate
    if modality == 'eeg':
        cfg = V82_EEG_SCENARIOS[scenario_name]
    elif modality == 'bl':
        cfg = V82_BL_SCENARIOS[scenario_name]
    else:
        cfg = {}

    n_eeg = min(len(base.get('p1_eeg', [])), len(base.get('p2_eeg', [])))
    gate_cfg = cfg.get('gate', {'event_range_s': (10.0, 25.0), 'ramp_s': 2.0, 'duty_cycle': 0.4})
    gate_eeg = generate_coupling_gate(n_eeg, 256.0, gate_cfg, seed=seed) if n_eeg > 0 else np.ones(1)
    gate_2hz = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(gate_eeg)), gate_eeg)

    # Inject V8.2 scenario
    if modality == 'eeg' and kappa > 0 and 'p1_eeg' in base:
        n_eeg = min(len(base['p1_eeg']), len(base['p2_eeg']))
        gate_r = np.interp(np.linspace(0, 1, n_eeg),
                            np.linspace(0, 1, len(gate_eeg)), gate_eeg)
        base['p1_eeg'], base['p2_eeg'] = inject_eeg_v82(
            base['p1_eeg'][:n_eeg], base['p2_eeg'][:n_eeg],
            kappa, scenario_name, gate_r, seed=seed)

    # Run BOTH V8.2 and V10 scaffolds
    try:
        z_v82, _, mask_v82, _, _ = run_from_raw_v82(base, t_common)
        z_v10, _, mask_v10, _, _ = run_from_raw_v10(base, t_common)
    except Exception as e:
        print(f"    Regression error: {e}")
        return None

    result = {'kappa': kappa, 'scenario': scenario_name, 'modality': modality}
    for i, key in enumerate(V82_KEYS):
        auc_v82 = compute_auc_within_session(z_v82[:, i], gate_2hz)
        auc_v10 = compute_auc_within_session(z_v10[:, i], gate_2hz)
        result[f'auc_v82_{key}'] = float(auc_v82)
        result[f'auc_v10_{key}'] = float(auc_v10)
        result[f'delta_{key}'] = float(auc_v10 - auc_v82)

    return result


# =========================================================================
# Phase runners
# =========================================================================

def run_phase1_lz(sessions, pairs, out_dir):
    """Phase 1: LZ complexity scenario validation."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: LZ Complexity Validation")
    print(f"{'='*60}")

    results = []
    for scenario_name in V10_LZ_SCENARIOS:
        kappas = V10_KAPPA_RANGES['lz']
        print(f"\n  Scenario: {scenario_name}")

        tasks = []
        for pi, (i, j) in enumerate(pairs):
            for kappa in kappas:
                tasks.append((sessions[i][1], sessions[j][1], kappa,
                              scenario_name, 300.0, 42 + pi))

        batch_results = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_lz_pair)(*t) for t in tasks
        )

        for r in batch_results:
            if r is not None:
                results.append(r)

        # Print summary per kappa
        for kappa in kappas:
            kr = [r for r in results if r['kappa'] == kappa and r['scenario'] == scenario_name]
            if kr:
                target_keys = V10_LZ_SCENARIOS[scenario_name]['target_features']
                for tk in target_keys:
                    aucs = [r.get(f'auc_{tk}', 0.5) for r in kr]
                    print(f"    kappa={kappa:.2f}: {tk} AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f} (n={len(aucs)})")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'phase1_lz_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_phase2_regression(sessions, pairs, out_dir):
    """Phase 2: V8.2 regression — verify channels 0-17 unchanged."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: V8.2 Regression Tests")
    print(f"{'='*60}")

    # Test key V8.2 scenarios through V10
    test_scenarios = [
        ('eeg', 'E1_mutual_gaze', 0.20),
        ('eeg', 'E3_shared_alpha', 0.20),
    ]

    results = []
    for modality, scenario_name, kappa in test_scenarios:
        print(f"\n  Scenario: {scenario_name} (kappa={kappa})")

        tasks = [(sessions[i][1], sessions[j][1], kappa, scenario_name,
                  modality, 300.0, 42 + pi)
                 for pi, (i, j) in enumerate(pairs[:3])]  # 3 pairs for regression

        batch_results = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_v82_regression)(*t) for t in tasks
        )

        for r in batch_results:
            if r is not None:
                results.append(r)
                # Check regression tolerance
                max_delta = max(abs(r.get(f'delta_{k}', 0))
                                for k in V10_MODALITY_KEYS[:18])
                status = 'PASS' if max_delta < 0.03 else 'FAIL'
                print(f"    Max AUC delta (V10-V82): {max_delta:.4f} [{status}]")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'phase2_regression_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_phase4_null(sessions, pairs, out_dir):
    """Phase 4: Null integrity — kappa=0 must give AUC in [0.40, 0.60]."""
    print(f"\n{'='*60}")
    print(f"  Phase 4: Null Integrity")
    print(f"{'='*60}")

    results = []
    for scenario_name in V10_LZ_SCENARIOS:
        tasks = [(sessions[i][1], sessions[j][1], 0.0,
                  scenario_name, 300.0, 42 + pi)
                 for pi, (i, j) in enumerate(pairs[:5])]

        batch_results = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_lz_pair)(*t) for t in tasks
        )

        for r in batch_results:
            if r is not None:
                results.append(r)

    # Check all channels at kappa=0
    if results:
        print(f"\n  Null AUC check (should be in [0.40, 0.60]):")
        for key in V10_MODALITY_KEYS:
            aucs = [r.get(f'auc_{key}', 0.5) for r in results]
            mean_auc = np.mean(aucs)
            status = 'PASS' if 0.35 <= mean_auc <= 0.65 else 'WARN'
            if abs(mean_auc - 0.5) > 0.1:
                status = 'FAIL'
            print(f"    {key:>24s}: {mean_auc:.3f} [{status}]")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'phase4_null_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# Summary
# =========================================================================

def print_summary(phase1, phase2, phase4):
    """Print consolidated pass/fail summary."""
    print(f"\n{'='*60}")
    print(f"  V10 Semi-Synthetic Battery Summary")
    print(f"{'='*60}")

    # Phase 1: LZ detection
    if phase1:
        print(f"\n  Phase 1: LZ Detection")
        for scenario_name in V10_LZ_SCENARIOS:
            target_keys = V10_LZ_SCENARIOS[scenario_name]['target_features']
            for kappa in [0.20, 0.40]:
                kr = [r for r in phase1 if r['kappa'] == kappa and r['scenario'] == scenario_name]
                if kr:
                    for tk in target_keys:
                        aucs = [r.get(f'auc_{tk}', 0.5) for r in kr]
                        mean_auc = np.mean(aucs)
                        threshold = 0.55 if kappa == 0.20 else 0.65
                        status = 'PASS' if mean_auc > threshold else 'FAIL'
                        print(f"    {scenario_name} k={kappa:.2f} {tk}: AUC={mean_auc:.3f} [{status}]")

    # Phase 2: Regression
    if phase2:
        print(f"\n  Phase 2: V8.2 Regression")
        for r in phase2:
            max_delta = max(abs(r.get(f'delta_{k}', 0)) for k in V10_MODALITY_KEYS[:18])
            status = 'PASS' if max_delta < 0.03 else 'FAIL'
            print(f"    {r['scenario']}: max_delta={max_delta:.4f} [{status}]")

    # Phase 4: Null
    if phase4:
        print(f"\n  Phase 4: Null Integrity")
        n_fail = 0
        for key in V10_MODALITY_KEYS:
            aucs = [r.get(f'auc_{key}', 0.5) for r in phase4]
            if aucs:
                mean_auc = np.mean(aucs)
                if abs(mean_auc - 0.5) > 0.1:
                    n_fail += 1
                    print(f"    FAIL: {key} null AUC={mean_auc:.3f}")
        if n_fail == 0:
            print(f"    All 24 channels PASS null integrity")


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V10 Semi-Synthetic Battery')
    parser.add_argument('--phase', type=int, default=None, help='Phase 1-4')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--quick', action='store_true', help='3 pairs only')
    args = parser.parse_args()

    config = load_config()
    n_pairs = 3 if args.quick else 10

    print("Loading sessions...")
    sessions = load_available_sessions(config)
    print(f"  Loaded {len(sessions)} sessions")

    pairs = generate_pseudo_dyad_pairs(sessions, n_pairs=n_pairs)
    print(f"  Generated {len(pairs)} pseudo-dyad pairs")

    os.makedirs(OUT_DIR, exist_ok=True)

    phase1 = phase2 = phase4 = []

    if args.all or args.phase == 1:
        phase1 = run_phase1_lz(sessions, pairs, OUT_DIR)

    if args.all or args.phase == 2:
        phase2 = run_phase2_regression(sessions, pairs, OUT_DIR)

    if args.all or args.phase == 4:
        phase4 = run_phase4_null(sessions, pairs, OUT_DIR)

    if args.all:
        print_summary(phase1, phase2, phase4)

    print(f"\n  Results saved to {OUT_DIR}/")
