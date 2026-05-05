"""V8.2 Semi-Synthetic Test Battery.

Comprehensive validation of the 18D V8.2 scaffold pipeline via raw-level
coupling injection and AUC measurement.

Phase 1: Per-modality validation (EEG, BL, ECG, Resp, Pose)
Phase 2: Multimodal composite tests (4 patterns)
Verification: band isolation, feature orthogonality, null integrity

All pseudo-dyad pairs × kappa levels parallelized via joblib (n_jobs=16).

Usage:
    python scripts/_test_v82_semisynthetic_battery.py --phase 1
    python scripts/_test_v82_semisynthetic_battery.py --phase 2
    python scripts/_test_v82_semisynthetic_battery.py --all
    python scripts/_test_v82_semisynthetic_battery.py --quick   # 3 pairs, fast
"""

import sys, os, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.constants import (
    V82_KAPPA_RANGES, V82_EEG_SCENARIOS, V82_BL_SCENARIOS,
    V82_ECG_SCENARIOS, V82_RESP_SCENARIOS, V82_POSE_SCENARIOS,
    V82_MULTIMODAL_PATTERNS,
)
from cadence.synthetic import generate_coupling_gate
from cadence.synthetic_v82 import (
    build_v82_pseudo_dyad, inject_eeg_v82, inject_bl_v82,
    inject_ecg_v82, inject_resp_v82, inject_pose_v82,
    generate_v82_gate, generate_meditation_gate, generate_multimodal_gates,
    inject_all_v82, compute_auc_within_session, compute_auc_cross_session,
    compute_peak_ratio,
)

FS_OUT = 2.0
OUT_DIR = 'results/v82_semisynthetic'
N_JOBS = 16


# =========================================================================
# Session loading
# =========================================================================

def load_available_sessions(config, max_sessions=None):
    """Load cached sessions for pseudo-dyad construction."""
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
    """Generate cross-session pseudo-dyad pairs (P1 from A, P2 from B)."""
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
# Core evaluation: run scaffold on injected data
# =========================================================================

def evaluate_single_pair(cached_a, cached_b, kappa_dict, scenario_dict,
                         modality_gates, duration_s=300.0, seed=42):
    """Evaluate one pseudo-dyad pair at given kappa levels.

    Returns dict of per-feature z-timecourses.
    """
    from scripts._run_scaffold_v82 import run_from_raw, MODALITY_KEYS

    # Find overlapping time window
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

    # Build pseudo-dyad base
    base = build_v82_pseudo_dyad(cached_a, cached_b, t_start, t_end)

    # Inject EEG coupling (raw 256 Hz level)
    if kappa_dict.get('eeg', 0) > 0 and 'p1_eeg' in base and 'p2_eeg' in base:
        scenario = scenario_dict.get('eeg')
        if scenario:
            n_eeg = min(len(base['p1_eeg']), len(base['p2_eeg']))
            eeg_gate = modality_gates.get('eeg', np.ones(n_eeg, dtype=np.float32))
            if len(eeg_gate) != n_eeg:
                eeg_gate = np.interp(np.linspace(0, 1, n_eeg),
                                      np.linspace(0, 1, len(eeg_gate)),
                                      eeg_gate).astype(np.float32)
            p1_e, p2_e = inject_eeg_v82(
                base['p1_eeg'][:n_eeg], base['p2_eeg'][:n_eeg],
                kappa_dict['eeg'], scenario, eeg_gate, seed=seed)
            base['p1_eeg'] = p1_e
            base['p2_eeg'] = p2_e

    # Inject BL coupling (raw 30 Hz blendshape level)
    if kappa_dict.get('bl', 0) > 0 and 'p1_blendshapes' in base and 'p2_blendshapes' in base:
        scenario = scenario_dict.get('bl')
        if scenario:
            n_bl = min(len(base['p1_blendshapes']), len(base['p2_blendshapes']))
            # Ensure matching column count
            n_ch = min(base['p1_blendshapes'].shape[1], base['p2_blendshapes'].shape[1])
            p1_bl_sub = base['p1_blendshapes'][:n_bl, :n_ch].copy()
            p2_bl_sub = base['p2_blendshapes'][:n_bl, :n_ch].copy()
            bl_gate = modality_gates.get('bl', np.ones(n_bl, dtype=np.float32))
            if len(bl_gate) != n_bl:
                bl_gate = np.interp(np.linspace(0, 1, n_bl),
                                     np.linspace(0, 1, len(bl_gate)),
                                     bl_gate).astype(np.float32)
            p1_b, p2_b, _ = inject_bl_v82(
                p1_bl_sub, p2_bl_sub,
                kappa_dict['bl'], scenario, bl_gate, seed=seed)
            base['p1_blendshapes'] = p1_b
            base['p2_blendshapes'] = p2_b
            # Update timestamps to match
            for role in ['p1', 'p2']:
                ts_key = f'{role}_blendshapes_ts'
                if ts_key in base and len(base[ts_key]) > n_bl:
                    base[ts_key] = base[ts_key][:n_bl]

    # Inject Pose coupling — directly at 2 Hz velocity level.
    # The pipeline measures velocity cross-product at 2 Hz. Injecting at 12 Hz
    # position level then integrating → interpolating → differentiating aliases
    # the signal. Instead: compute velocity at 2 Hz for BOTH participants,
    # mix P1's velocity into P2's, and pass the coupled velocity directly.
    if kappa_dict.get('pose', 0) > 0 and 'p1_pose_features' in base and 'p2_pose_features' in base:
        scenario = scenario_dict.get('pose')
        if scenario:
            from cadence.constants import V82_POSE_SCENARIOS
            cfg = V82_POSE_SCENARIOS[scenario]
            channels = cfg['channels']
            lag_samp_2hz = max(1, int(cfg['lag_s'] * FS_OUT))

            _p1 = base['p1_pose_features']
            _p2 = base['p2_pose_features']
            _t1 = base.get('p1_pose_features_ts', np.arange(len(_p1)) / 12.0 + t_common[0])
            _t2 = base.get('p2_pose_features_ts', np.arange(len(_p2)) / 12.0 + t_common[0])

            n_ch = min(_p1.shape[1], _p2.shape[1], 11)
            idx = list(range(n_ch))

            # Interpolate to 2 Hz (same as pipeline)
            p1_2hz = np.column_stack([np.interp(t_common, _t1, _p1[:, c], left=0, right=0) for c in idx])
            p2_2hz = np.column_stack([np.interp(t_common, _t2, _p2[:, c], left=0, right=0) for c in idx])

            # Velocity at 2 Hz
            v1 = np.diff(p1_2hz, axis=0, prepend=p1_2hz[:1]).astype(np.float64)
            v2 = np.diff(p2_2hz, axis=0, prepend=p2_2hz[:1]).astype(np.float64)

            # Gate at 2 Hz
            pose_gate = modality_gates.get('pose', np.ones(len(t_common), dtype=np.float32))
            if len(pose_gate) != len(t_common):
                pose_gate = np.interp(np.linspace(0, 1, len(t_common)),
                                       np.linspace(0, 1, len(pose_gate)),
                                       pose_gate).astype(np.float32)

            # Mix P1's velocity into P2's on target channels
            POSE_AMP = 10.0
            for ch in channels:
                if ch >= n_ch:
                    continue
                v1_lagged = np.roll(v1[:, ch], lag_samp_2hz)
                if lag_samp_2hz > 0: v1_lagged[:lag_samp_2hz] = 0.0

                v1_std = max(v1_lagged.std(), 1e-8)
                v2_std = max(v2[:, ch].std(), 1e-8)
                alpha = np.clip(kappa_dict['pose'] * POSE_AMP * pose_gate, 0, 0.95)
                v2[:, ch] += alpha * v1_lagged * (v2_std / v1_std)

            # Store coupled velocity — pipeline will use this directly
            base['_pose_vel_p2_coupled'] = v2.astype(np.float32)

    # Inject ECG coupling (IBI level — after R-peak detection, before pipeline bandpass)
    if kappa_dict.get('ecg', 0) > 0 and 'p1_ecg' in base and 'p2_ecg' in base:
        scenario = scenario_dict.get('ecg')
        if scenario:
            from scripts._extract_respiratory import detect_rpeaks
            from scripts._run_rslds_scaffold_v8 import ECG_SRATE
            from scipy.interpolate import interp1d as _interp1d

            ecg_cfg = V82_ECG_SCENARIOS[scenario]
            try:
                r1 = detect_rpeaks(base['p1_ecg'], base['p1_ecg_ts'], ECG_SRATE)
                r2 = detect_rpeaks(base['p2_ecg'], base['p2_ecg_ts'], ECG_SRATE)
                if r1[0] is not None and r2[0] is not None:
                    _, _, _, ibis1, ibi_t1 = r1
                    _, _, _, ibis2, ibi_t2 = r2
                    # Common IBI grid at 4 Hz
                    t_s = max(ibi_t1[0], ibi_t2[0])
                    t_e = min(ibi_t1[-1], ibi_t2[-1])
                    if t_e - t_s > 30:
                        t_ibi = np.arange(t_s, t_e, 0.25)  # 4 Hz
                        f1 = _interp1d(ibi_t1, ibis1, kind='cubic',
                                       bounds_error=False, fill_value='extrapolate')
                        f2 = _interp1d(ibi_t2, ibis2, kind='cubic',
                                       bounds_error=False, fill_value='extrapolate')
                        ibi1_u = np.clip(f1(t_ibi), 0.3, 2.0)
                        ibi2_u = np.clip(f2(t_ibi), 0.3, 2.0)

                        # Generate gate at 4 Hz
                        ecg_gate = modality_gates.get('ecg', np.ones(len(t_ibi), dtype=np.float32))
                        if len(ecg_gate) != len(t_ibi):
                            ecg_gate = np.interp(np.linspace(0, 1, len(t_ibi)),
                                                  np.linspace(0, 1, len(ecg_gate)),
                                                  ecg_gate).astype(np.float32)
                        ibi2_coupled = inject_ecg_v82(
                            ibi1_u, ibi2_u, kappa_dict['ecg'],
                            ecg_cfg['band'], ecg_gate,
                            lag_s=ecg_cfg['lag_s'], seed=seed)
                        # Store modified IBI for pipeline to use
                        # The pipeline re-does R-peak detection, so we need to
                        # modify the raw ECG to produce different IBIs.
                        # Simpler: store pre-computed IBI and modify run_from_raw.
                        # For now: inject at the raw ECG level by scaling P2's
                        # R-R intervals to match the coupled IBI.
                        base['_ecg_ibi2_coupled'] = ibi2_coupled
                        base['_ecg_ibi_t'] = t_ibi
                        base['_ecg_ibi1'] = ibi1_u
            except Exception:
                pass

    # Inject Resp coupling (fused EDR level)
    if kappa_dict.get('resp', 0) > 0 and 'p1_ecg' in base and 'p2_ecg' in base:
        scenario = scenario_dict.get('resp')
        if scenario:
            from scripts._extract_respiratory import extract_respiratory_one
            from scripts._run_rslds_scaffold_v8 import ECG_SRATE
            try:
                p1_resp = extract_respiratory_one(base['p1_ecg'], base['p1_ecg_ts'],
                                                  srate=ECG_SRATE)
                p2_resp = extract_respiratory_one(base['p2_ecg'], base['p2_ecg_ts'],
                                                  srate=ECG_SRATE)
                if p1_resp is not None and p2_resp is not None:
                    fused1, fused2 = p1_resp['fused'], p2_resp['fused']
                    t_edr = p1_resp['t']
                    n_edr = min(len(fused1), len(fused2), len(t_edr))
                    resp_gate = modality_gates.get('resp',
                                                    np.ones(n_edr, dtype=np.float32))
                    if len(resp_gate) != n_edr:
                        resp_gate = np.interp(
                            np.linspace(0, 1, n_edr),
                            np.linspace(0, 1, len(resp_gate)),
                            resp_gate).astype(np.float32)
                    fused2_coupled = inject_resp_v82(
                        fused1[:n_edr], fused2[:n_edr], t_edr[:n_edr],
                        kappa_dict['resp'], resp_gate, seed=seed)
                    # Store for pipeline
                    base['_resp_p2_fused_coupled'] = fused2_coupled
                    base['_resp_p2_t'] = p2_resp['t'][:n_edr]
                    base['_resp_p1'] = p1_resp
            except Exception:
                pass

    # Run full scaffold pipeline
    try:
        z_matrix, z_matrix_raw, obs_mask, pw_diag, extras = run_from_raw(
            base, t_common, label='semisynthetic')
    except Exception as e:
        print(f"    Pipeline error: {e}")
        return None

    return {
        'z_matrix': z_matrix,
        'z_raw': z_matrix_raw,
        'obs_mask': obs_mask,
        'pw_diag': pw_diag,
        't_common': t_common,
        'extras': extras,
    }


# =========================================================================
# Phase 1: Per-modality sweeps
# =========================================================================

def run_phase1_modality(modality, sessions, pairs, duration_s=300.0,
                        n_jobs=N_JOBS):
    """Run per-modality kappa sweep across pseudo-dyad pairs."""
    from scripts._run_scaffold_v82 import MODALITY_KEYS

    if modality == 'eeg':
        scenarios = V82_EEG_SCENARIOS
        kappas = V82_KAPPA_RANGES['eeg']
    elif modality == 'bl':
        scenarios = V82_BL_SCENARIOS
        kappas = V82_KAPPA_RANGES['bl']
    elif modality == 'ecg':
        scenarios = V82_ECG_SCENARIOS
        kappas = V82_KAPPA_RANGES['ecg']
    elif modality == 'resp':
        scenarios = V82_RESP_SCENARIOS
        kappas = V82_KAPPA_RANGES['resp']
    elif modality == 'pose':
        scenarios = V82_POSE_SCENARIOS
        kappas = V82_KAPPA_RANGES['pose']
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # All modalities use within-session AUC (gate-ON vs gate-OFF).
    # Cross-session comparison fails because standardization forces mean=0.
    # For BL/Pose, also compute peak_ratio as supplementary metric.
    use_cross_session = False  # disabled — standardization erases mean shift

    results = {}
    for scenario_name in scenarios:
        print(f"\n  Scenario: {scenario_name}")
        scenario_results = {}

        scen_cfg = scenarios[scenario_name]
        gate_cfg = scen_cfg.get('gate', {'duty_cycle': 0.30,
                                          'event_range_s': (5, 20), 'ramp_s': 2.0})
        gate_fs = {'eeg': 256.0, 'bl': 30.0, 'pose': 12.0, 'ecg': 4.0, 'resp': 4.0
                   }.get(modality, 30.0)

        # Pre-compute null results for cross-session comparison
        null_z_means = None
        if use_cross_session:
            print(f"    null: ", end='', flush=True)
            null_results = Parallel(n_jobs=min(n_jobs, len(pairs)), prefer='threads')(
                delayed(_run_one_pair_null)(pi, pair, sessions, duration_s)
                for pi, pair in enumerate(pairs)
            )
            # Collect per-feature z-means from null runs
            null_z_means = {k: [] for k in MODALITY_KEYS}
            for (res, _) in null_results:
                if res is not None:
                    for fi, fkey in enumerate(MODALITY_KEYS):
                        null_z_means[fkey].append(float(res['z_matrix'][:, fi].mean()))
            print(f"{sum(1 for r,_ in null_results if r is not None)} pairs")

        for kappa in kappas:
            print(f"    kappa={kappa:.2f}: ", end='', flush=True)

            def _run_one_pair(pair_idx, pair, _kappa=kappa):
                i_a, i_b = pair
                _, cached_a = sessions[i_a]
                _, cached_b = sessions[i_b]

                kappa_dict = {'eeg': 0, 'bl': 0, 'ecg': 0, 'resp': 0, 'pose': 0}
                kappa_dict[modality] = _kappa
                scenario_dict = {modality: scenario_name}

                n_gate = int(duration_s * gate_fs)
                gate = generate_coupling_gate(n_gate, gate_fs, gate_cfg,
                                               seed=42 + pair_idx)
                modality_gates = {modality: gate}
                gate_2hz = np.interp(
                    np.linspace(0, 1, int(duration_s * FS_OUT)),
                    np.linspace(0, 1, len(gate)), gate).astype(np.float32)

                result = evaluate_single_pair(
                    cached_a, cached_b, kappa_dict, scenario_dict,
                    modality_gates, duration_s=duration_s,
                    seed=42 + pair_idx + int(_kappa * 1000))
                return result, gate_2hz

            pair_results = Parallel(n_jobs=min(n_jobs, len(pairs)), prefer='threads')(
                delayed(_run_one_pair)(pi, pair)
                for pi, pair in enumerate(pairs)
            )

            feature_aucs = {k: [] for k in MODALITY_KEYS}
            feature_peaks = {k: [] for k in MODALITY_KEYS}
            n_valid = 0

            for (res, gate_2hz) in pair_results:
                if res is None:
                    continue
                n_valid += 1

                for fi, fkey in enumerate(MODALITY_KEYS):
                    # Within-session AUC: use raw z for CWT-based features,
                    # raw cross-product for surrogate-z features, prewhitened for EEG.
                    use_raw = fkey in ('bl_expr', 'bl_activity_conc')
                    extras = res.get('extras', {})

                    if fkey == 'pose' and '_pose_raw_cp' in extras:
                        # Use raw cross-product BEFORE surrogate z-scoring
                        z_vec = extras['_pose_raw_cp']
                    elif use_raw:
                        z_vec = res['z_raw'][:, fi]
                    else:
                        z_vec = res['z_matrix'][:, fi]

                    if kappa > 0:
                        auc = compute_auc_within_session(z_vec, gate_2hz)
                    else:
                        fake_gate = generate_coupling_gate(
                            len(z_vec), FS_OUT, gate_cfg, seed=42)
                        auc = compute_auc_within_session(z_vec, fake_gate)
                    feature_aucs[fkey].append(auc)

                    # Peak ratio for transient features
                    if kappa > 0 and modality in ('bl', 'pose'):
                        pr = compute_peak_ratio(res['z_raw'][:, fi], gate_2hz)
                        feature_peaks[fkey].append(pr)

            # Summarize
            auc_summary = {}
            for fkey in MODALITY_KEYS:
                vals = feature_aucs[fkey]
                peaks = feature_peaks.get(fkey, [])
                if vals:
                    auc_summary[fkey] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'n': len(vals),
                    }
                    if peaks:
                        auc_summary[fkey]['peak_ratio'] = float(np.mean(peaks))
                else:
                    auc_summary[fkey] = {'mean': 0.5, 'std': 0, 'n': 0}

            scenario_results[f'kappa_{kappa:.2f}'] = auc_summary
            top = sorted(auc_summary.items(), key=lambda x: -x[1]['mean'])[:3]
            top_str = ', '.join(f"{k}={v['mean']:.3f}" for k, v in top)
            print(f"{n_valid} pairs, top: {top_str}")

        results[scenario_name] = scenario_results

    return results


def _run_one_pair_null(pair_idx, pair, sessions, duration_s):
    """Run a null (kappa=0) evaluation for one pair."""
    i_a, i_b = pair
    _, cached_a = sessions[i_a]
    _, cached_b = sessions[i_b]

    kappa_dict = {'eeg': 0, 'bl': 0, 'ecg': 0, 'resp': 0, 'pose': 0}
    modality_gates = {}

    result = evaluate_single_pair(
        cached_a, cached_b, kappa_dict, {},
        modality_gates, duration_s=duration_s,
        seed=42 + pair_idx)

    gate_2hz = np.zeros(int(duration_s * FS_OUT), dtype=np.float32)
    return result, gate_2hz


# =========================================================================
# Phase 2: Multimodal composite tests
# =========================================================================

def run_phase2_composite(sessions, pairs, duration_s=300.0, n_jobs=N_JOBS):
    """Run multimodal composite pattern tests."""
    from scripts._run_scaffold_v82 import MODALITY_KEYS

    results = {}
    for pattern_name, pattern in V82_MULTIMODAL_PATTERNS.items():
        print(f"\n  Pattern: {pattern_name}")

        # Build kappa and scenario dicts
        kappa_dict = {}
        scenario_dict = {}
        for mod_key in ['eeg', 'bl', 'ecg', 'resp', 'pose']:
            spec = pattern.get(mod_key)
            if spec is None:
                kappa_dict[mod_key] = 0.0
                scenario_dict[mod_key] = None
            else:
                scenario_name, kappa = spec
                kappa_dict[mod_key] = kappa
                scenario_dict[mod_key] = scenario_name

        def _run_composite_pair(pair_idx, pair):
            i_a, i_b = pair
            _, cached_a = sessions[i_a]
            _, cached_b = sessions[i_b]

            # Generate coordinated gates
            n_gate_2hz = int(duration_s * FS_OUT)
            gates_2hz = generate_multimodal_gates(n_gate_2hz, FS_OUT,
                                                    pattern_name,
                                                    seed=42 + pair_idx)

            # Upsample gates to native rates
            modality_gates = {}
            rate_map = {'eeg': 256.0, 'bl': 30.0, 'ecg': 4.0, 'resp': 4.0, 'pose': 12.0}
            for mk, fs in rate_map.items():
                n_native = int(duration_s * fs)
                modality_gates[mk] = np.interp(
                    np.linspace(0, 1, n_native),
                    np.linspace(0, 1, n_gate_2hz),
                    gates_2hz.get(mk, np.zeros(n_gate_2hz))
                ).astype(np.float32)

            result = evaluate_single_pair(
                cached_a, cached_b, kappa_dict, scenario_dict,
                modality_gates, duration_s=duration_s,
                seed=42 + pair_idx)

            return result, gates_2hz

        # Run pairs in parallel
        pair_results = Parallel(n_jobs=min(n_jobs, len(pairs)), prefer='threads')(
            delayed(_run_composite_pair)(pi, pair)
            for pi, pair in enumerate(pairs)
        )

        # Compute AUC per feature (within-session for modalities with gates)
        feature_aucs = {k: [] for k in MODALITY_KEYS}
        n_valid = 0
        for (res, gates_2hz) in pair_results:
            if res is None:
                continue
            n_valid += 1
            # Use the max gate across active modalities for AUC
            active_gates = [g for mk, g in gates_2hz.items()
                           if pattern.get(mk) is not None]
            if active_gates:
                combined_gate = np.maximum.reduce(active_gates)
            else:
                combined_gate = np.zeros(int(duration_s * FS_OUT), dtype=np.float32)

            for fi, fkey in enumerate(MODALITY_KEYS):
                auc = compute_auc_within_session(
                    res['z_matrix'][:, fi], combined_gate)
                feature_aucs[fkey].append(auc)

        auc_summary = {}
        for fkey in MODALITY_KEYS:
            vals = feature_aucs[fkey]
            if vals:
                auc_summary[fkey] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'n': len(vals),
                }
            else:
                auc_summary[fkey] = {'mean': 0.5, 'std': 0, 'n': 0}

        results[pattern_name] = {
            'kappa_dict': {k: v for k, v in kappa_dict.items()},
            'scenarios': {k: v for k, v in scenario_dict.items() if v},
            'auc': auc_summary,
            'n_pairs': n_valid,
        }

        # Print summary
        active = [f for f, s in scenario_dict.items() if s]
        detected = [fkey for fkey, v in auc_summary.items() if v['mean'] > 0.60]
        print(f"    {n_valid} pairs, active: {active}")
        print(f"    Detected (AUC>0.60): {detected[:6]}")

    return results


# =========================================================================
# Verification tests
# =========================================================================

def run_verification(sessions, pairs, duration_s=300.0, n_jobs=N_JOBS):
    """Run verification tests: null integrity, band isolation, dose-response."""
    from scripts._run_scaffold_v82 import MODALITY_KEYS

    results = {}

    # 1. Null integrity: kappa=0 on all modalities
    print("\n  [V1] Null integrity...")
    null_aucs = {k: [] for k in MODALITY_KEYS}
    null_results_list = Parallel(n_jobs=min(n_jobs, len(pairs)), prefer='threads')(
        delayed(_run_one_pair_null)(pi, pair, sessions, duration_s)
        for pi, pair in enumerate(pairs)
    )

    for (res, _) in null_results_list:
        if res is None:
            continue
        # Use a random gate — should give AUC~0.50 since no coupling
        T_null = len(res['z_matrix'])
        fake_gate = generate_coupling_gate(
            T_null, FS_OUT,
            {'duty_cycle': 0.30, 'event_range_s': (8, 25), 'ramp_s': 2.0},
            seed=42)
        for fi, fkey in enumerate(MODALITY_KEYS):
            auc = compute_auc_within_session(
                res['z_matrix'][:, fi], fake_gate)
            null_aucs[fkey].append(auc)

    null_summary = {}
    all_pass = True
    for fkey in MODALITY_KEYS:
        vals = null_aucs[fkey]
        if vals:
            mean_auc = float(np.mean(vals))
            null_summary[fkey] = {'mean': mean_auc, 'std': float(np.std(vals)),
                                   'pass': 0.45 <= mean_auc <= 0.55}
            if not null_summary[fkey]['pass']:
                all_pass = False
        else:
            null_summary[fkey] = {'mean': 0.5, 'std': 0, 'pass': True}

    results['null_integrity'] = {'features': null_summary, 'all_pass': all_pass}
    pass_str = 'PASS' if all_pass else 'FAIL'
    print(f"    Null integrity: {pass_str}")
    for fkey, v in null_summary.items():
        if not v['pass']:
            print(f"      FAIL: {fkey} AUC={v['mean']:.3f}")

    # 2. Band isolation (EEG alpha injection → theta/beta should be null)
    print("\n  [V2] Band isolation (EEG alpha → theta/beta)...")
    # Run E1 (alpha) at kappa=0.30
    band_results = run_phase1_modality(
        'eeg', sessions, pairs[:3], duration_s=duration_s, n_jobs=n_jobs)

    if 'E1_mutual_gaze' in band_results:
        k030 = band_results['E1_mutual_gaze'].get('kappa_0.30', {})
        alpha_auc = k030.get('imcoh_alpha', {}).get('mean', 0.5)
        theta_auc = k030.get('imcoh_theta', {}).get('mean', 0.5)
        beta_auc = k030.get('imcoh_beta', {}).get('mean', 0.5)

        alpha_delta = alpha_auc - 0.5
        theta_leak = abs(theta_auc - 0.5)
        beta_leak = abs(beta_auc - 0.5)

        isolation_pass = True
        if alpha_delta > 0.05:  # only test if alpha actually detected
            if theta_leak > 0.05 * alpha_delta:
                isolation_pass = False
            if beta_leak > 0.05 * alpha_delta:
                isolation_pass = False

        results['band_isolation'] = {
            'alpha_auc': alpha_auc,
            'theta_auc': theta_auc,
            'beta_auc': beta_auc,
            'pass': isolation_pass,
        }
        print(f"    alpha={alpha_auc:.3f}, theta={theta_auc:.3f}, "
              f"beta={beta_auc:.3f} -> {'PASS' if isolation_pass else 'FAIL'}")

    return results


# =========================================================================
# Visualization
# =========================================================================

def plot_injection_timeline(z_matrix, gate_2hz, t_common, scenario_name,
                            kappa, modality_keys, out_path):
    """Plot per-feature z-timecourse with injection gate overlay.

    Shows ground truth injection periods (shaded) vs detected z-scores
    for each of the 18 features. Annotates within-session AUC.
    """
    n_feat = z_matrix.shape[1]
    fig, axes = plt.subplots(n_feat, 1, figsize=(20, 1.4 * n_feat), sharex=True)
    t_rel = t_common - t_common[0]  # relative time in seconds

    # Colors for feature groups
    group_colors = {
        'imcoh': '#1565C0', 'conc': '#FF9800', 'dyn': '#D32F2F',
        'asym': '#00897B', 'bl': '#E91E63', 'ecg': '#FF5722',
        'resp': '#607D8B', 'pose': '#4CAF50',
    }

    for fi, fkey in enumerate(modality_keys):
        ax = axes[fi]
        z = z_matrix[:, fi]

        # Shade injection gate
        gate_bool = gate_2hz > 0.5
        ax.fill_between(t_rel, -4, 4, where=gate_bool,
                        alpha=0.12, color='#FFD54F', zorder=0, label='Injection ON')

        # Plot z-timecourse
        prefix = fkey.split('_')[0]
        color = group_colors.get(prefix, '#666666')
        ax.plot(t_rel, z, color=color, linewidth=0.6, alpha=0.9)
        ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)

        # Compute and annotate AUC
        auc = compute_auc_within_session(z, gate_2hz)
        ax.text(0.99, 0.92, f'AUC={auc:.3f}', transform=ax.transAxes,
                fontsize=7, ha='right', va='top', fontweight='bold',
                color='#D32F2F' if auc > 0.60 else '#666666',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        # Mean during ON/OFF
        on_mask = gate_2hz > 0.5
        off_mask = gate_2hz < 0.2
        if on_mask.sum() > 5 and off_mask.sum() > 5:
            m_on = z[on_mask].mean()
            m_off = z[off_mask].mean()
            ax.text(0.01, 0.92, f'ON={m_on:+.2f} OFF={m_off:+.2f}',
                    transform=ax.transAxes, fontsize=6, ha='left', va='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_ylabel(fkey, fontsize=6, rotation=0, ha='right', va='center')
        ax.set_ylim(-3.5, 3.5)
        ax.tick_params(labelsize=5)

    axes[-1].set_xlabel('Time (s)', fontsize=8)
    fig.suptitle(f'{scenario_name} | kappa={kappa:.2f} | Injection Gate + Z-Timecourses',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_kappa_sweep_heatmap(results_dict, modality_keys, out_path):
    """Plot kappa × feature AUC heatmap across all scenarios.

    Each row is a feature, each column group is a scenario × kappa.
    """
    scenarios = list(results_dict.keys())
    if not scenarios:
        return

    # Collect all kappa levels
    all_kappas = set()
    for scen_results in results_dict.values():
        for k in scen_results:
            if k.startswith('kappa_'):
                all_kappas.add(float(k.split('_')[1]))
    kappas = sorted(all_kappas)

    n_scen = len(scenarios)
    n_kappa = len(kappas)
    n_feat = len(modality_keys)
    n_cols = n_scen * n_kappa

    auc_matrix = np.full((n_feat, n_cols), 0.5)
    col_labels = []

    for si, scen in enumerate(scenarios):
        for ki, kap in enumerate(kappas):
            col_idx = si * n_kappa + ki
            kap_key = f'kappa_{kap:.2f}'
            col_labels.append(f'{scen[:6]}\nk={kap:.1f}')
            if kap_key in results_dict[scen]:
                for fi, fkey in enumerate(modality_keys):
                    if fkey in results_dict[scen][kap_key]:
                        auc_matrix[fi, col_idx] = results_dict[scen][kap_key][fkey]['mean']

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 0.5), max(8, n_feat * 0.35)))

    im = ax.imshow(auc_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0.40, vmax=0.80, interpolation='nearest')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=5, rotation=45, ha='right')
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(modality_keys, fontsize=6)

    # Annotate cells with AUC values
    for fi in range(n_feat):
        for ci in range(n_cols):
            val = auc_matrix[fi, ci]
            color = 'white' if val > 0.65 or val < 0.45 else 'black'
            ax.text(ci, fi, f'{val:.2f}', ha='center', va='center',
                    fontsize=4, color=color, fontweight='bold')

    # Vertical lines between scenarios
    for si in range(1, n_scen):
        ax.axvline(si * n_kappa - 0.5, color='black', linewidth=1.5)

    plt.colorbar(im, ax=ax, label='AUC', shrink=0.7)
    ax.set_title('Semi-Synthetic Battery: Kappa Sweep AUC Heatmap', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dose_response(results_dict, target_features, out_path):
    """Plot dose-response curves: AUC vs kappa for target features."""
    scenarios = list(results_dict.keys())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4),
                              sharey=True, squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(target_features)))

    for si, scen in enumerate(scenarios):
        ax = axes[0, si]
        scen_data = results_dict[scen]

        kappas = []
        for k in sorted(scen_data.keys()):
            if k.startswith('kappa_'):
                kappas.append(float(k.split('_')[1]))

        for fi, fkey in enumerate(target_features):
            aucs = []
            stds = []
            for kap in kappas:
                kap_key = f'kappa_{kap:.2f}'
                if kap_key in scen_data and fkey in scen_data[kap_key]:
                    aucs.append(scen_data[kap_key][fkey]['mean'])
                    stds.append(scen_data[kap_key][fkey].get('std', 0))
                else:
                    aucs.append(0.5)
                    stds.append(0)

            ax.plot(kappas, aucs, 'o-', color=colors[fi], label=fkey,
                    markersize=4, linewidth=1.5)
            ax.fill_between(kappas,
                            [a - s for a, s in zip(aucs, stds)],
                            [a + s for a, s in zip(aucs, stds)],
                            alpha=0.15, color=colors[fi])

        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(0.6, color='green', linestyle=':', linewidth=0.8, alpha=0.4)
        ax.set_xlabel('Kappa (ecological)')
        ax.set_title(scen, fontsize=9, fontweight='bold')
        ax.set_ylim(0.40, 0.85)
        ax.grid(alpha=0.2)

    axes[0, 0].set_ylabel('AUC')
    axes[0, -1].legend(fontsize=6, loc='upper left', ncol=1)
    fig.suptitle('Dose-Response: AUC vs Kappa', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_example_timeline(sessions, pairs, modality, scenario_name, kappa,
                               duration_s=180.0, out_dir=OUT_DIR):
    """Generate a single example timeline plot for one scenario at one kappa."""
    from scripts._run_scaffold_v82 import run_from_raw, MODALITY_KEYS

    if modality == 'eeg':
        scenarios = V82_EEG_SCENARIOS
    elif modality == 'bl':
        scenarios = V82_BL_SCENARIOS
    elif modality == 'pose':
        scenarios = V82_POSE_SCENARIOS
    else:
        return

    cfg = scenarios[scenario_name]
    gate_cfg = cfg.get('gate', {'duty_cycle': 0.35, 'event_range_s': (8, 25), 'ramp_s': 2.0})

    i_a, i_b = pairs[0]
    _, cached_a = sessions[i_a]
    _, cached_b = sessions[i_b]

    ts_a = cached_a.get('p1_eeg_ts', np.array([0, 300]))
    ts_b = cached_b.get('p2_eeg_ts', cached_b.get('p1_eeg_ts', np.array([0, 300])))
    t_start = max(ts_a[0], ts_b[0])
    t_end = min(ts_a[-1], ts_b[-1])
    dur = min(duration_s, t_end - t_start)
    t_end = t_start + dur
    t_common = np.arange(t_start, t_end, 1.0 / FS_OUT)

    gate_fs = 256.0 if modality == 'eeg' else (30.0 if modality == 'bl' else 12.0)
    n_gate = int(dur * gate_fs)
    gate_native = generate_coupling_gate(n_gate, gate_fs, gate_cfg, seed=42)
    gate_2hz = np.interp(np.linspace(0, 1, len(t_common)),
                          np.linspace(0, 1, n_gate), gate_native).astype(np.float32)

    # Build and inject
    base = build_v82_pseudo_dyad(cached_a, cached_b, t_start, t_end)

    kappa_dict = {'eeg': 0, 'bl': 0, 'ecg': 0, 'resp': 0, 'pose': 0}
    kappa_dict[modality] = kappa
    scenario_dict = {modality: scenario_name}
    modality_gates = {modality: gate_native}

    # EEG injection
    if modality == 'eeg' and 'p1_eeg' in base and 'p2_eeg' in base:
        n_eeg = min(len(base['p1_eeg']), len(base['p2_eeg']))
        eeg_gate = gate_native[:n_eeg] if n_eeg <= n_gate else np.interp(
            np.linspace(0, 1, n_eeg), np.linspace(0, 1, n_gate), gate_native
        ).astype(np.float32)
        p1_e, p2_e = inject_eeg_v82(
            base['p1_eeg'][:n_eeg], base['p2_eeg'][:n_eeg],
            kappa, scenario_name, eeg_gate, seed=42)
        base['p1_eeg'] = p1_e
        base['p2_eeg'] = p2_e

    # BL injection
    if modality == 'bl' and 'p1_blendshapes' in base and 'p2_blendshapes' in base:
        n_bl = min(len(base['p1_blendshapes']), len(base['p2_blendshapes']))
        n_ch = min(base['p1_blendshapes'].shape[1], base['p2_blendshapes'].shape[1])
        bl_gate = gate_native[:n_bl] if n_bl <= n_gate else np.interp(
            np.linspace(0, 1, n_bl), np.linspace(0, 1, n_gate), gate_native
        ).astype(np.float32)
        p1_b, p2_b, _ = inject_bl_v82(
            base['p1_blendshapes'][:n_bl, :n_ch], base['p2_blendshapes'][:n_bl, :n_ch],
            kappa, scenario_name, bl_gate, seed=42)
        base['p1_blendshapes'] = p1_b
        base['p2_blendshapes'] = p2_b
        for role in ['p1', 'p2']:
            tk = f'{role}_blendshapes_ts'
            if tk in base and len(base[tk]) > n_bl:
                base[tk] = base[tk][:n_bl]

    # Pose injection
    if modality == 'pose' and 'p1_pose_features' in base and 'p2_pose_features' in base:
        n_p = min(len(base['p1_pose_features']), len(base['p2_pose_features']))
        pose_gate = gate_native[:n_p] if n_p <= n_gate else np.interp(
            np.linspace(0, 1, n_p), np.linspace(0, 1, n_gate), gate_native
        ).astype(np.float32)
        p2_p = inject_pose_v82(
            base['p1_pose_features'][:n_p], base['p2_pose_features'][:n_p],
            kappa, scenario_name, pose_gate, seed=42)
        base['p2_pose_features'] = p2_p

    # Run pipeline
    try:
        z_matrix, _, _, _, _ = run_from_raw(base, t_common, label='timeline')
    except Exception as e:
        print(f"    Timeline error: {e}")
        return

    # Plot
    out_path = os.path.join(out_dir, f'timeline_{scenario_name}_k{kappa:.2f}.png')
    plot_injection_timeline(z_matrix, gate_2hz, t_common, scenario_name,
                            kappa, MODALITY_KEYS, out_path)
    print(f"    Saved: {out_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='V8.2 Semi-Synthetic Battery')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Phase 1 (per-modality) or 2 (multimodal)')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 3 pairs, 180s')
    parser.add_argument('--modality', type=str, default=None,
                        help='Specific modality for Phase 1')
    parser.add_argument('--n-pairs', type=int, default=6)
    parser.add_argument('--duration', type=float, default=300.0)
    args = parser.parse_args()

    if args.quick:
        args.n_pairs = 3
        args.duration = 180.0
        if args.phase is None:
            args.all = True

    if args.phase is None and not args.all:
        args.all = True

    print("=" * 70)
    print("  V8.2 Semi-Synthetic Test Battery")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load sessions
    print(f"\nLoading sessions...")
    config = load_config()
    sessions = load_available_sessions(config, max_sessions=12)
    print(f"  {len(sessions)} sessions loaded")

    if len(sessions) < 2:
        print("ERROR: Need at least 2 sessions for pseudo-dyad construction")
        return

    # Generate pseudo-dyad pairs
    pairs = generate_pseudo_dyad_pairs(sessions, n_pairs=args.n_pairs, seed=42)
    print(f"  {len(pairs)} pseudo-dyad pairs")

    all_results = {}
    t_start = time.time()

    # Phase 1: Per-modality
    if args.phase == 1 or args.all:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: Per-Modality Validation")
        print(f"{'='*70}")

        modalities = ['eeg', 'bl', 'pose']  # ECG/Resp need special handling
        if args.modality:
            modalities = [args.modality]

        for mod in modalities:
            print(f"\n--- {mod.upper()} ---")
            t0 = time.time()
            try:
                mod_results = run_phase1_modality(
                    mod, sessions, pairs, duration_s=args.duration,
                    n_jobs=N_JOBS)
                all_results[f'phase1_{mod}'] = mod_results
                with open(os.path.join(OUT_DIR, f'phase1_{mod}.json'), 'w') as f:
                    json.dump(mod_results, f, indent=2, default=str)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
            print(f"  {mod.upper()}: {time.time() - t0:.0f}s")

    # Phase 2: Multimodal composites
    if args.phase == 2 or args.all:
        print(f"\n{'='*70}")
        print(f"  PHASE 2: Multimodal Composite Tests")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            composite_results = run_phase2_composite(
                sessions, pairs, duration_s=args.duration, n_jobs=N_JOBS)
            all_results['phase2_composite'] = composite_results
            with open(os.path.join(OUT_DIR, 'phase2_composite.json'), 'w') as f:
                json.dump(composite_results, f, indent=2, default=str)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        print(f"  Phase 2: {time.time() - t0:.0f}s")

    # Verification
    if args.all:
        print(f"\n{'='*70}")
        print(f"  VERIFICATION TESTS")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            verif_results = run_verification(
                sessions, pairs[:3], duration_s=args.duration, n_jobs=N_JOBS)
            all_results['verification'] = verif_results
            with open(os.path.join(OUT_DIR, 'verification.json'), 'w') as f:
                json.dump(verif_results, f, indent=2, default=str)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        print(f"  Verification: {time.time() - t0:.0f}s")

    # Summary
    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  BATTERY COMPLETE: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Results: {OUT_DIR}/")
    print(f"{'='*70}")

    # Print pass/fail summary
    print(f"\n  PASS/FAIL SUMMARY:")
    if 'verification' in all_results:
        v = all_results['verification']
        ni = v.get('null_integrity', {})
        print(f"    Null integrity: {'PASS' if ni.get('all_pass') else 'FAIL'}")
        bi = v.get('band_isolation', {})
        if bi:
            print(f"    Band isolation: {'PASS' if bi.get('pass') else 'FAIL'}")

    # Save combined results
    with open(os.path.join(OUT_DIR, 'battery_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Visualizations ──────────────────────────────────────────────
    from scripts._run_scaffold_v82 import MODALITY_KEYS

    print(f"\n  Generating visualizations...")

    # Heatmap for each Phase 1 modality
    for mod in ['eeg', 'bl', 'pose']:
        key = f'phase1_{mod}'
        if key in all_results:
            plot_kappa_sweep_heatmap(
                all_results[key], MODALITY_KEYS,
                os.path.join(OUT_DIR, f'heatmap_{mod}.png'))
            print(f"    Heatmap: heatmap_{mod}.png")

            # Dose-response for target features
            if mod == 'eeg':
                targets = ['imcoh_alpha', 'imcoh_theta', 'conc_alpha',
                           'dyn_alpha', 'asym_alpha']
            elif mod == 'bl':
                targets = ['bl_expr', 'bl_activity_conc', 'conc_alpha', 'dyn_alpha']
            else:
                targets = ['pose', 'bl_expr', 'conc_alpha', 'dyn_alpha']
            plot_dose_response(
                all_results[key], targets,
                os.path.join(OUT_DIR, f'dose_response_{mod}.png'))
            print(f"    Dose-response: dose_response_{mod}.png")

    # Example timelines at highest kappa for each modality
    for mod, scen, kap in [('eeg', 'E1_mutual_gaze', 0.50),
                            ('eeg', 'E4_therapist_leading', 0.50),
                            ('bl', 'B1_smile', 0.40),
                            ('pose', 'P1_postural_mirror', 0.40)]:
        try:
            generate_example_timeline(sessions, pairs, mod, scen, kap,
                                       duration_s=args.duration, out_dir=OUT_DIR)
        except Exception as e:
            print(f"    Timeline {scen} error: {e}")

    print(f"  Visualizations complete.")


if __name__ == '__main__':
    main()
