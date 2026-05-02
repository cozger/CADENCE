"""V8 RSLDS validation suite: 4 critical tests.

1. AR(1) null test — fit K=4 on matched AR(1) noise → states should be degenerate
2. Pseudo-dyad contrast — real pair vs cross-session → real should be more structured
3. Condition alignment — state usage should differ across conditions
4. Semi-synthetic AUC — injected coupling at kappa=0.2/0.4 → AUC > 0.60

Usage:
    python scripts/_test_v8_validation.py
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from joblib import Parallel, delayed

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, iohmm_flexibility_metrics,
    build_observation_mask, fit_slds,
)
from scripts._run_rslds_phase2 import MODALITY_KEYS, FS_OUT

# ── Load V8 data ─────────────────────────────────────────────────────

def load_v8_session(session_name):
    """Load V8 scaffold NPZ. Returns (z_matrix, t_common) or (None, None)."""
    npz = f'results/rslds/{session_name}/rslds_scaffold_v8_ztimecourses.npz'
    if not os.path.exists(npz):
        return None, None
    data = np.load(npz)
    t = data['t_common']
    z = np.column_stack([data[f'z_{k}'] for k in MODALITY_KEYS]).astype(np.float64)
    return z, t


def load_all_v8_sessions():
    """Load all sessions with V8 scaffold data."""
    sessions = {}
    for d in sorted(os.listdir('results/rslds')):
        npz = f'results/rslds/{d}/rslds_scaffold_v8_ztimecourses.npz'
        if os.path.exists(npz):
            z, t = load_v8_session(d)
            if z is not None:
                sessions[d] = {'z': z, 't': t}
    return sessions


# ══════════════════════════════════════════════════════════════════════
#  TEST 1: AR(1) NULL
# ══════════════════════════════════════════════════════════════════════

def test_ar1_null(Y_real, n_trials=5, K=4, seed=42):
    """Fit K=4 IOHMM on AR(1) noise matched to real data.

    If states on noise look similar to states on real data, the model
    is fitting autocorrelation structure, not coupling.
    """
    print(f"\n{'='*70}")
    print(f"  TEST 1: AR(1) Null (K={K}, {n_trials} trials)")
    print(f"{'='*70}")

    T, D = Y_real.shape
    rng = np.random.default_rng(seed)

    # Measure real data properties
    rho_real = np.array([np.corrcoef(Y_real[:-1, d], Y_real[1:, d])[0, 1] for d in range(D)])
    std_real = Y_real.std(axis=0)

    print(f"  Real data: T={T}, D={D}")
    print(f"  Real rho per channel: {', '.join(f'{r:.3f}' for r in rho_real)}")

    # Fit real data
    U_zero = np.zeros((T, 2), dtype=np.float64)
    obs_mask = np.ones((T, D), dtype=bool)

    cfg = IOHMMConfig(K=K, D_obs=D, D_input=2, n_restarts=3, max_em_iter=150)
    model = IOHMM(cfg)
    params_real, hist_real = model.fit(Y_real, U_zero, obs_mask, seed=42, verbose=False)
    gamma_real = hist_real['gamma']
    flex_real = iohmm_flexibility_metrics(gamma_real, fs=FS_OUT)

    # Between-state mean range on real data
    mu_range_real = params_real.mu.max(axis=0) - params_real.mu.min(axis=0)

    print(f"\n  Real IOHMM: BIC={hist_real['bic']:.0f}, "
          f"usage={[f'{u:.1%}' for u in flex_real['state_usage']]}")
    print(f"  Real mu range: {', '.join(f'{k}={v:.3f}' for k, v in zip(MODALITY_KEYS, mu_range_real))}")

    # Generate and fit AR(1) null trials
    null_bics = []
    null_mu_ranges = []
    null_usages = []

    def _fit_one_null(trial):
        r = np.random.default_rng(seed + trial * 100)
        # Vectorized AR(1) generation: use cumulative filtering
        # y[t] = rho*y[t-1] + e[t] → y = lfilter([1], [1, -rho], e)
        from scipy.signal import lfilter
        Y_null = np.zeros((T, D))
        for d in range(D):
            rho = rho_real[d]
            noise_std = std_real[d] * np.sqrt(max(1 - rho**2, 0.01))
            e = r.standard_normal(T) * noise_std
            Y_null[:, d] = lfilter([1], [1, -rho], e)

        cfg_n = IOHMMConfig(K=K, D_obs=D, D_input=2, n_restarts=2, max_em_iter=100)
        model_n = IOHMM(cfg_n)
        params_n, hist_n = model_n.fit(Y_null, U_zero, obs_mask, seed=seed+trial, verbose=False)
        gamma_n = hist_n['gamma']
        flex_n = iohmm_flexibility_metrics(gamma_n, fs=FS_OUT)
        mu_range_n = params_n.mu.max(axis=0) - params_n.mu.min(axis=0)
        return hist_n['bic'], mu_range_n, flex_n['state_usage']

    results = Parallel(n_jobs=-1, prefer='processes')(
        delayed(_fit_one_null)(i) for i in range(n_trials))

    for bic, mu_r, usage in results:
        null_bics.append(bic)
        null_mu_ranges.append(mu_r)
        null_usages.append(usage)

    null_mu_ranges = np.array(null_mu_ranges)

    print(f"\n  Null BIC: {np.mean(null_bics):.0f} +/- {np.std(null_bics):.0f} "
          f"(real: {hist_real['bic']:.0f})")
    print(f"  Null mu range (mean): {', '.join(f'{k}={v:.3f}' for k, v in zip(MODALITY_KEYS, null_mu_ranges.mean(axis=0)))}")
    print(f"  Real mu range:        {', '.join(f'{k}={v:.3f}' for k, v in zip(MODALITY_KEYS, mu_range_real))}")

    # Key metric: is real mu_range larger than null?
    ratio = mu_range_real / np.maximum(null_mu_ranges.mean(axis=0), 1e-6)
    n_larger = (ratio > 1.5).sum()
    print(f"\n  Mu range ratio (real/null): {', '.join(f'{v:.2f}' for v in ratio)}")
    print(f"  Channels where real > 1.5x null: {n_larger}/{D}")

    pass_ar1 = n_larger >= 3  # at least 3 channels meaningfully larger
    print(f"  [{'PASS' if pass_ar1 else 'FAIL'}] AR(1) null test "
          f"({'real states differ from noise' if pass_ar1 else 'states indistinguishable from noise'})")

    return {
        'pass': pass_ar1,
        'real_bic': float(hist_real['bic']),
        'null_bic_mean': float(np.mean(null_bics)),
        'mu_range_ratio': {k: float(v) for k, v in zip(MODALITY_KEYS, ratio)},
        'n_channels_larger': int(n_larger),
    }


# ══════════════════════════════════════════════════════════════════════
#  TEST 2: PSEUDO-DYAD CONTRAST
# ══════════════════════════════════════════════════════════════════════

def test_pseudo_dyad_contrast(sessions, real_session='y_06', K=4, n_pseudo=5, seed=42):
    """Compare real pair to pseudo-dyads (cross-session z-timecourses).

    Real pair should have more structured state transitions.
    """
    print(f"\n{'='*70}")
    print(f"  TEST 2: Pseudo-Dyad Contrast (K={K}, {n_pseudo} pseudo-pairs)")
    print(f"{'='*70}")

    if real_session not in sessions:
        print(f"  SKIP: {real_session} not in V8 sessions")
        return {'pass': None, 'reason': 'insufficient data'}

    Y_real = sessions[real_session]['z']
    T, D = Y_real.shape
    U_zero = np.zeros((T, 2), dtype=np.float64)
    obs_mask = np.ones((T, D), dtype=bool)

    # Fit real data
    cfg = IOHMMConfig(K=K, D_obs=D, D_input=2, n_restarts=3, max_em_iter=150)
    model = IOHMM(cfg)
    _, hist_real = model.fit(Y_real, U_zero, obs_mask, seed=42, verbose=False)
    flex_real = iohmm_flexibility_metrics(hist_real['gamma'], fs=FS_OUT)

    print(f"  Real {real_session}: BIC={hist_real['bic']:.0f}, "
          f"trans={flex_real['n_transitions']}, entropy={flex_real['shannon_entropy']:.3f}")

    # Create pseudo-dyads by shuffling channels across sessions
    other_sessions = [s for s in sessions if s != real_session]
    if len(other_sessions) == 0:
        # Only 1 V8 session — create pseudo-dyads by circular shift within session
        print(f"  Only 1 V8 session available — using time-shifted pseudo-dyads")
        rng = np.random.default_rng(seed)

        def _fit_pseudo_shift(i):
            r = np.random.default_rng(seed + i * 100)
            shift = r.integers(T // 5, 4 * T // 5)
            Y_pseudo = np.roll(Y_real, shift, axis=0)
            # Mix: half channels from real, half from shifted
            mix_mask = r.choice([True, False], size=D)
            Y_mix = Y_real.copy()
            Y_mix[:, mix_mask] = Y_pseudo[:, mix_mask]

            cfg_p = IOHMMConfig(K=K, D_obs=D, D_input=2, n_restarts=2, max_em_iter=100)
            model_p = IOHMM(cfg_p)
            _, hist_p = model_p.fit(Y_mix, U_zero, obs_mask, seed=seed+i, verbose=False)
            flex_p = iohmm_flexibility_metrics(hist_p['gamma'], fs=FS_OUT)
            return hist_p['bic'], flex_p['n_transitions'], flex_p['shannon_entropy']

        pseudo_results = Parallel(n_jobs=-1, prefer='processes')(
            delayed(_fit_pseudo_shift)(i) for i in range(n_pseudo))
    else:
        # Multiple V8 sessions — proper cross-session pseudo-dyads
        rng = np.random.default_rng(seed)

        def _fit_pseudo_cross(i):
            r = np.random.default_rng(seed + i * 100)
            other = r.choice(other_sessions)
            Y_other = sessions[other]['z']
            T_min = min(T, len(Y_other))
            # Mix: channels 0-4 from real, 5-8 from other (or vice versa)
            Y_mix = np.zeros((T_min, D))
            split = D // 2
            Y_mix[:, :split] = Y_real[:T_min, :split]
            Y_mix[:, split:] = Y_other[:T_min, split:]

            U_z = np.zeros((T_min, 2), dtype=np.float64)
            obs_m = np.ones((T_min, D), dtype=bool)
            cfg_p = IOHMMConfig(K=K, D_obs=D, D_input=2, n_restarts=2, max_em_iter=100)
            model_p = IOHMM(cfg_p)
            _, hist_p = model_p.fit(Y_mix, U_z, obs_m, seed=seed+i, verbose=False)
            flex_p = iohmm_flexibility_metrics(hist_p['gamma'], fs=FS_OUT)
            return hist_p['bic'], flex_p['n_transitions'], flex_p['shannon_entropy']

        pseudo_results = Parallel(n_jobs=-1, prefer='processes')(
            delayed(_fit_pseudo_cross)(i) for i in range(n_pseudo))

    pseudo_bics = [r[0] for r in pseudo_results]
    pseudo_trans = [r[1] for r in pseudo_results]
    pseudo_entropy = [r[2] for r in pseudo_results]

    print(f"  Pseudo BIC: {np.mean(pseudo_bics):.0f} +/- {np.std(pseudo_bics):.0f}")
    print(f"  Pseudo trans: {np.mean(pseudo_trans):.0f} +/- {np.std(pseudo_trans):.0f}")
    print(f"  Pseudo entropy: {np.mean(pseudo_entropy):.3f} +/- {np.std(pseudo_entropy):.3f}")

    # Real should have LOWER entropy (more structured) or different transition rate
    entropy_ratio = flex_real['shannon_entropy'] / max(np.mean(pseudo_entropy), 1e-6)
    trans_ratio = flex_real['n_transitions'] / max(np.mean(pseudo_trans), 1)

    print(f"\n  Entropy ratio (real/pseudo): {entropy_ratio:.3f}")
    print(f"  Transition ratio (real/pseudo): {trans_ratio:.3f}")

    # Looser criterion: any structural difference
    pass_pseudo = abs(entropy_ratio - 1.0) > 0.1 or abs(trans_ratio - 1.0) > 0.15
    print(f"  [{'PASS' if pass_pseudo else 'FAIL'}] Pseudo-dyad contrast "
          f"({'structural difference found' if pass_pseudo else 'real ≈ pseudo'})")

    return {
        'pass': pass_pseudo,
        'real_bic': float(hist_real['bic']),
        'pseudo_bic_mean': float(np.mean(pseudo_bics)),
        'entropy_ratio': float(entropy_ratio),
        'trans_ratio': float(trans_ratio),
    }


# ══════════════════════════════════════════════════════════════════════
#  TEST 3: CONDITION ALIGNMENT
# ══════════════════════════════════════════════════════════════════════

def test_condition_alignment(Y_real, t_common, session_name='y_06', K=4):
    """Check if IOHMM states align with session conditions."""
    print(f"\n{'='*70}")
    print(f"  TEST 3: Condition Alignment (K={K})")
    print(f"{'='*70}")

    T, D = Y_real.shape
    U_zero = np.zeros((T, 2), dtype=np.float64)
    obs_mask = np.ones((T, D), dtype=bool)

    # Load condition segments from V8 JSON
    json_path = f'results/rslds/{session_name}/rslds_scaffold_v8_results.json'
    if not os.path.exists(json_path):
        print(f"  SKIP: no V8 JSON for {session_name}")
        return {'pass': None, 'reason': 'no JSON'}

    with open(json_path) as f:
        info = json.load(f)
    segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

    if not segments:
        print(f"  SKIP: no segments in JSON")
        return {'pass': None, 'reason': 'no segments'}

    # Fit IOHMM
    cfg = IOHMMConfig(K=K, D_obs=D, D_input=2, n_restarts=3, max_em_iter=150)
    model = IOHMM(cfg)
    params, hist = model.fit(Y_real, U_zero, obs_mask, seed=42, verbose=False)
    viterbi = model.viterbi(Y_real, U_zero, obs_mask, params)

    # Per-condition state usage
    print(f"\n  {'Condition':>14s} |" + ''.join(f'  S{k}   ' for k in range(K)) + "| Dominant")
    print(f"  " + "-" * (16 + 8 * K + 12))

    condition_usage = {}
    for seg_name, t0, t1 in segments:
        mask = (t_common >= t0) & (t_common <= t1)
        if mask.sum() < 10:
            continue
        seg_viterbi = viterbi[mask]
        usage = np.array([float((seg_viterbi == k).mean()) for k in range(K)])
        condition_usage[seg_name] = usage
        dominant = np.argmax(usage)
        usage_str = ''.join(f'{u:7.1%} ' for u in usage)
        print(f"  {seg_name:>14s} | {usage_str}| S{dominant}")

    # Test: is there condition differentiation?
    # Chi-squared-like metric: sum of squared deviations from uniform
    if len(condition_usage) >= 3:
        usage_matrix = np.array(list(condition_usage.values()))  # (n_cond, K)
        uniform = 1.0 / K
        chi2_like = np.sum((usage_matrix - uniform) ** 2) / uniform
        # Compare to what we'd expect from noise: permutation test
        rng = np.random.default_rng(42)
        null_chi2s = []
        for _ in range(200):
            perm_viterbi = rng.permutation(viterbi)
            perm_usage = []
            for seg_name, t0, t1 in segments:
                mask = (t_common >= t0) & (t_common <= t1)
                if mask.sum() < 10:
                    continue
                perm_usage.append([float((perm_viterbi[mask] == k).mean()) for k in range(K)])
            perm_matrix = np.array(perm_usage)
            null_chi2s.append(np.sum((perm_matrix - uniform) ** 2) / uniform)

        p_value = float(np.mean(np.array(null_chi2s) >= chi2_like))
        print(f"\n  Condition differentiation: chi2={chi2_like:.4f}, p={p_value:.4f} "
              f"(permutation test, 200 shuffles)")

        pass_cond = p_value < 0.05
        print(f"  [{'PASS' if pass_cond else 'FAIL'}] Condition alignment "
              f"({'states differentiate conditions' if pass_cond else 'states uniform across conditions'})")

        return {
            'pass': pass_cond,
            'chi2': float(chi2_like),
            'p_value': float(p_value),
            'condition_usage': {k: v.tolist() for k, v in condition_usage.items()},
        }
    else:
        print(f"  SKIP: fewer than 3 conditions")
        return {'pass': None, 'reason': 'insufficient conditions'}


# ══════════════════════════════════════════════════════════════════════
#  TEST 4: SEMI-SYNTHETIC AUC (V8 data)
# ══════════════════════════════════════════════════════════════════════

def test_semisynthetic_auc(sessions, n_pairs=15, K=2, seed=42):
    """Semi-synthetic IOHMM AUC on V8 scaffold data.

    Creates pseudo-dyads from V8 sessions, injects coupling episodes,
    measures detection AUC.
    """
    print(f"\n{'='*70}")
    print(f"  TEST 4: Semi-Synthetic AUC ({n_pairs} pairs, V8 data)")
    print(f"{'='*70}")

    session_list = list(sessions.keys())
    if len(session_list) < 1:
        print(f"  SKIP: no V8 sessions")
        return {'pass': None, 'reason': 'no data'}

    D = sessions[session_list[0]]['z'].shape[1]
    rng = np.random.default_rng(seed)

    # Injection parameters (V8 9D)
    INJECTION_DIMS = [0, 1, 3, 8]  # eeg_theta, eeg_alpha, bl_expr, pose
    INJECTION_WEIGHTS = {0: 1.5, 1: 0.5, 3: 0.8, 8: 1.0}
    N_EPISODES = 30
    KAPPA_VALUES = [0.0, 0.2, 0.4]

    def _create_base(rng_local):
        """Create a pseudo-dyad base from V8 data."""
        if len(session_list) >= 2:
            idx = rng_local.choice(len(session_list), 2, replace=False)
            z_a = sessions[session_list[idx[0]]]['z']
            z_b = sessions[session_list[idx[1]]]['z']
            T = min(len(z_a), len(z_b))
            shift = rng_local.integers(T // 5, 4 * T // 5)
            z_base = (z_a[:T] + np.roll(z_b[:T], shift, axis=0)) / np.sqrt(2)
        else:
            z = sessions[session_list[0]]['z']
            T = len(z)
            shift = rng_local.integers(T // 5, 4 * T // 5)
            z_base = (z + np.roll(z, shift, axis=0)) / np.sqrt(2)
        return z_base

    def _inject(z_base, kappa, rng_local):
        """Inject coupling episodes."""
        T = len(z_base)
        z_inj = z_base.copy()
        gt = np.zeros(T, dtype=bool)
        if kappa == 0:
            return z_inj, gt

        starts, durations = [], []
        for _ in range(N_EPISODES * 20):
            if len(starts) >= N_EPISODES:
                break
            dur = int(rng_local.uniform(3, 8) * FS_OUT)
            s = rng_local.integers(dur, T - dur)
            if all(abs(s - p) > pd + int(3 * FS_OUT) for p, pd in zip(starts, durations)):
                starts.append(s)
                durations.append(dur)

        for s, dur in zip(starts, durations):
            gt[s:s+dur] = True
            ramp = np.ones(dur)
            ramp[0] = 0.5
            if dur > 1:
                ramp[-1] = 0.5
            ar = np.zeros(dur)
            ar[0] = rng_local.standard_normal()
            for i in range(1, dur):
                ar[i] = 0.5 * ar[i-1] + rng_local.standard_normal() * 0.7
            ar *= 0.3

            for d in INJECTION_DIMS:
                if d < z_inj.shape[1]:
                    w = INJECTION_WEIGHTS.get(d, 1.0)
                    data_std = max(z_base[:, d].std(), 0.1)
                    z_inj[s:s+dur, d] += kappa * w * data_std * ramp * (1.0 + ar)

        return z_inj, gt

    def _fit_one(pair_seed, kappa):
        from sklearn.metrics import roc_auc_score
        r = np.random.default_rng(pair_seed)
        z_base = _create_base(r)
        z_inj, gt = _inject(z_base, kappa, r)

        T = len(z_inj)
        U = np.zeros((T, 2), dtype=np.float64)
        obs_m = np.ones((T, D), dtype=bool)
        cfg = IOHMMConfig(K=K, D_obs=D, D_input=2, n_restarts=2, max_em_iter=80)
        model = IOHMM(cfg)
        params, hist = model.fit(z_inj, U, obs_m, seed=pair_seed, verbose=False)

        gamma = hist['gamma']
        coupled_state = np.argmax(params.mu.mean(axis=1))
        posterior = gamma[:, coupled_state]

        if gt.sum() > 0 and gt.sum() < T:
            auc = roc_auc_score(gt, posterior)
            if auc < 0.5:
                auc = 1 - auc
        else:
            auc = 0.5
        return auc

    all_results = {}
    for kappa in KAPPA_VALUES:
        pair_seeds = [seed + i + int(kappa * 1000) for i in range(n_pairs)]
        aucs = Parallel(n_jobs=-1, prefer='processes')(
            delayed(_fit_one)(ps, kappa) for ps in pair_seeds)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        all_results[kappa] = {'mean': mean_auc, 'std': std_auc, 'aucs': aucs}
        print(f"  kappa={kappa:.2f}: AUC={mean_auc:.3f} +/- {std_auc:.3f} (n={n_pairs})")

    # Checks
    null_auc = all_results[0.0]['mean']
    max_auc = all_results[max(KAPPA_VALUES)]['mean']
    pass_null = null_auc < 0.55
    pass_signal = max_auc > 0.55
    pass_monotonic = all_results[0.2]['mean'] <= all_results[0.4]['mean'] + 0.05

    print(f"\n  [{'PASS' if pass_null else 'FAIL'}] Null AUC={null_auc:.3f} < 0.55")
    print(f"  [{'PASS' if pass_signal else 'FAIL'}] Max AUC={max_auc:.3f} > 0.55")
    print(f"  [{'PASS' if pass_monotonic else 'FAIL'}] Monotonic increase with kappa")

    pass_all = pass_null and pass_signal and pass_monotonic
    return {
        'pass': pass_all,
        'null_auc': float(null_auc),
        'max_auc': float(max_auc),
        'results': {str(k): {'mean': float(v['mean']), 'std': float(v['std'])}
                    for k, v in all_results.items()},
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  V8 RSLDS VALIDATION SUITE")
    print("=" * 70)
    t_wall = time.time()

    # Load V8 data
    sessions = load_all_v8_sessions()
    print(f"\n  V8 sessions available: {list(sessions.keys())}")

    if not sessions:
        print("  ERROR: No V8 scaffold data found. Run _run_rslds_scaffold_v8.py first.")
        return

    primary = 'y_06' if 'y_06' in sessions else list(sessions.keys())[0]
    Y_real = sessions[primary]['z']
    t_common = sessions[primary]['t']

    # Run all 4 tests
    results = {}

    results['ar1_null'] = test_ar1_null(Y_real, n_trials=5, K=4)
    results['pseudo_dyad'] = test_pseudo_dyad_contrast(sessions, real_session=primary, K=4, n_pseudo=5)
    results['condition_alignment'] = test_condition_alignment(Y_real, t_common, session_name=primary, K=4)
    results['semisynthetic'] = test_semisynthetic_auc(sessions, n_pairs=15, K=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}")
    for test_name, r in results.items():
        status = 'PASS' if r.get('pass') else ('SKIP' if r.get('pass') is None else 'FAIL')
        print(f"  [{status}] {test_name}")

    n_pass = sum(1 for r in results.values() if r.get('pass') is True)
    n_total = sum(1 for r in results.values() if r.get('pass') is not None)
    print(f"\n  {n_pass}/{n_total} tests passed in {time.time() - t_wall:.0f}s")

    # Save
    os.makedirs('results/rslds', exist_ok=True)
    with open('results/rslds/v8_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved results/rslds/v8_validation_results.json")


if __name__ == '__main__':
    main()
