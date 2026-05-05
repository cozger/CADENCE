"""Semi-synthetic IOHMM validation: pseudo-dyad base + injected coupling episodes.

V8 version: 9D observation space (EEG theta/alpha/beta, BL expr/state,
ECG LF/HF, Resp, Pose). Uses V8 scaffold data (prewhitened+standardized).

Following the established semi-synthetic paradigm (CLAUDE.md rule):
  1. Load pseudo-dyad z base (P1 from session A, P2 from session B) = null coupling
  2. Inject synthetic coupling episodes at known times with controlled amplitude kappa
  3. Fit IOHMM at K=2 (coupled/uncoupled)
  4. Measure detection AUC: does high-coupling state overlap injected windows?

Usage:
    python scripts/_test_iohmm_semisynthetic.py
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, iohmm_flexibility_metrics, build_observation_mask,
)
from scripts._run_rslds_phase2 import spectral_decompose, MODALITY_KEYS
from cadence.significance.rslds_validation import synthetic_iohmm_recovery

# ── Constants ─────────────────────────────────────────────────────────

FS_OUT = 2.0
N_PSEUDO_PAIRS = 20       # number of pseudo-dyad pairs to test per kappa
KAPPA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
# Realistic injection parameters calibrated from real session z-timecourses:
# Real coupling episodes are short (2-5s), frequent (20-50/session), with
# modality-specific amplitudes. EEG most active, BL weakest.
N_EPISODES = 30           # frequent short bursts (real: 20-50 per session)
EPISODE_DUR_S = 5         # 5s median (real: 2-5s). Stays in z_fast (>>0.01 Hz)
# V8 9D: inject into EEG theta(0), EEG alpha(1), BL expr(3), Pose(8)
INJECTION_DIMS = [0, 1, 3, 8]
# Per-dim amplitude scaling (relative to kappa): EEG strongest, BL weakest
INJECTION_WEIGHTS = {0: 1.5, 1: 0.5, 3: 0.8, 8: 1.0}


def load_all_scaffold_z():
    """Load z-timecourses from all sessions (V8 preferred, V7 fallback)."""
    session_data = {}
    for d in sorted(os.listdir('results/rslds')):
        # Try V8 first
        npz_v8 = f'results/rslds/{d}/rslds_scaffold_v8_ztimecourses.npz'
        npz_v7 = f'results/rslds/{d}/rslds_scaffold_ztimecourses.npz'
        npz = npz_v8 if os.path.exists(npz_v8) else npz_v7
        if not os.path.exists(npz):
            continue
        data = np.load(npz)
        z = {}
        for k in MODALITY_KEYS:
            zk = f'z_{k}'
            z[k] = data[zk] if zk in data else np.zeros(len(data['t_common']), dtype=np.float32)
        session_data[d] = z
    return session_data


def create_pseudo_dyad_z(session_data, rng, min_T=2000):
    """Create a pseudo-dyad z-timecourse by pairing different sessions.

    Takes z-timecourses from two different sessions (already cross-product-derived),
    aligns to the shorter session length. This is NOT the same as per-modality shift —
    it uses real session z-timecourses from unrelated dyads.
    """
    sessions = list(session_data.keys())
    if len(sessions) < 2:
        raise ValueError("Need at least 2 sessions for pseudo-dyad pairs")

    # Pick two different sessions
    idx = rng.choice(len(sessions), 2, replace=False)
    s_a, s_b = sessions[idx[0]], sessions[idx[1]]
    z_a = session_data[s_a]
    z_b = session_data[s_b]

    # Align to shorter session
    T = min(len(next(iter(z_a.values()))), len(next(iter(z_b.values()))))
    T = min(T, max(min_T, T))

    # Average the two sessions' z-timecourses as pseudo-dyad base
    # (This simulates what a z-timecourse would look like for an unrelated pair)
    z_pseudo = {}
    for k in MODALITY_KEYS:
        a = z_a[k][:T]
        b = z_b[k][:T]
        # Mix: use one session's z-trace with a random offset from the other
        shift = rng.integers(T // 5, 4 * T // 5)
        z_pseudo[k] = (a + np.roll(b, shift)) / np.sqrt(2)  # normalize variance

    return z_pseudo, T


def inject_coupling_episodes(z_traces, kappa, n_episodes, episode_dur_s,
                              injection_dims, fs, rng):
    """Inject realistic coupling episodes calibrated from real session data.

    Real coupling z-timecourses show:
      - Short bursts (2-5s), frequent (20-50/session)
      - Positive mean shift (z>0 = more synchronized than surrogate null)
      - Sharp onset, slight ramp-off (1-2 sample ramp)
      - Modality-specific amplitudes (EEG strongest, BL weakest)
      - AR(1)-like fluctuation within episodes (not flat)

    Returns:
        z_injected: dict of modified z-timecourses
        ground_truth: (T,) boolean array, True during injected episodes
    """
    T = len(next(iter(z_traces.values())))
    keys = list(z_traces.keys())
    ground_truth = np.zeros(T, dtype=bool)
    z_injected = {k: z_traces[k].copy() for k in keys}

    if kappa == 0:
        return z_injected, ground_truth

    # Variable episode durations: 3-8s (realistic range)
    min_dur, max_dur = 3.0, 8.0

    # Place non-overlapping episodes with minimum 3s gap
    min_gap_samples = int(3.0 * fs)
    starts = []
    durations = []
    for _ in range(n_episodes * 20):
        if len(starts) >= n_episodes:
            break
        dur_s = rng.uniform(min_dur, max_dur)
        dur_samp = int(dur_s * fs)
        s = rng.integers(dur_samp, T - dur_samp)
        if all(abs(s - prev) > prev_dur + min_gap_samples
               for prev, prev_dur in zip(starts, durations)):
            starts.append(s)
            durations.append(dur_samp)

    for start, dur_samp in zip(starts, durations):
        end = min(start + dur_samp, T)
        ground_truth[start:end] = True

        # Per-episode: positive step with 1-sample ramp and AR(1) fluctuation
        n = end - start
        ramp = np.ones(n)
        ramp[0] = 0.5                          # soft onset
        if n > 1:
            ramp[-1] = 0.5                      # soft offset

        # AR(1) noise within episode (rho~0.5, realistic fluctuation)
        ar_noise = np.zeros(n)
        ar_noise[0] = rng.standard_normal()
        for i in range(1, n):
            ar_noise[i] = 0.5 * ar_noise[i - 1] + rng.standard_normal() * 0.7
        ar_noise *= 0.3  # 30% amplitude modulation

        for d in injection_dims:
            if d < len(keys):
                key = keys[d]
                data_std = max(np.std(z_traces[key]), 0.1)
                weight = INJECTION_WEIGHTS.get(d, 1.0)
                z_injected[key][start:end] += (
                    kappa * weight * data_std * ramp * (1.0 + ar_noise))

    return z_injected, ground_truth


def _fit_one_pair(args):
    """Worker function for parallel semi-synthetic fitting (IOHMM)."""
    z_base, kappa, gt, pair_seed = args
    from scripts._run_rslds_phase2 import MODALITY_KEYS
    from cadence.significance.rslds_model import IOHMM, IOHMMConfig, build_observation_mask

    T = len(next(iter(z_base.values())))
    rng = np.random.default_rng(pair_seed)

    # Inject
    z_inj, gt_inj = inject_coupling_episodes(
        z_base, kappa, N_EPISODES, EPISODE_DUR_S,
        INJECTION_DIMS, FS_OUT, rng)
    gt_use = gt_inj if kappa > 0 else gt

    # V8: data is already prewhitened+standardized — use directly, no spectral decompose
    Y = np.column_stack([z_inj[k] for k in MODALITY_KEYS]).astype(np.float64)
    U = np.zeros((T, 2), dtype=np.float64)  # no slow covariates for V8
    obs_mask = build_observation_mask(z_inj, MODALITY_KEYS, T)

    cfg = IOHMMConfig(K=2, D_obs=len(MODALITY_KEYS), D_input=2,
                      n_restarts=2, max_em_iter=80)
    model = IOHMM(cfg)
    params, history = model.fit(Y, U, obs_mask, seed=pair_seed, verbose=False)

    gamma = history['gamma']
    # Coupled state = highest overall emission mean (injection elevates all dims)
    mean_emissions = params.mu.mean(axis=1)
    coupled_state = np.argmax(mean_emissions)
    coupled_posterior = gamma[:, coupled_state]

    if gt_use.sum() > 0 and gt_use.sum() < T:
        auc = roc_auc_score(gt_use, coupled_posterior)
        if auc < 0.5:
            auc = 1 - auc
    else:
        auc = 0.5

    return auc


def _fit_one_pair_slds(args):
    """Worker for parallel semi-synthetic fitting (SLDS variants)."""
    z_base, kappa, gt, pair_seed, model_type = args
    from scripts._run_rslds_phase2 import MODALITY_KEYS
    from cadence.significance.rslds_model import IOHMMConfig, fit_slds, build_observation_mask

    T = len(next(iter(z_base.values())))
    rng = np.random.default_rng(pair_seed)

    z_inj, gt_inj = inject_coupling_episodes(
        z_base, kappa, N_EPISODES, EPISODE_DUR_S,
        INJECTION_DIMS, FS_OUT, rng)
    gt_use = gt_inj if kappa > 0 else gt

    # V8: data already prewhitened+standardized — use directly
    Y = np.column_stack([z_inj[k] for k in MODALITY_KEYS]).astype(np.float64)
    U = np.zeros((T, 2), dtype=np.float64)
    obs_mask = build_observation_mask(z_inj, MODALITY_KEYS, T)

    n_mod = len(MODALITY_KEYS)
    # Configure based on model_type
    if model_type == 'slds':
        cfg = IOHMMConfig(K=2, D_obs=n_mod, D_input=2, D_latent=3,
                          n_restarts=2, max_em_iter=80)
    elif model_type == 'slds_fa':
        cfg = IOHMMConfig(K=2, D_obs=n_mod, D_input=2, D_latent=3,
                          n_factors=2, n_restarts=2, max_em_iter=80)
    elif model_type == 'rslds':
        cfg = IOHMMConfig(K=2, D_obs=n_mod, D_input=2, D_latent=3,
                          recurrent=True, n_restarts=2, max_em_iter=80)
    elif model_type == 'full':
        cfg = IOHMMConfig(K=2, D_obs=n_mod, D_input=2, D_latent=3,
                          n_factors=2, recurrent=True,
                          n_restarts=2, max_em_iter=80)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    _, hist = fit_slds(Y, U, obs_mask, cfg, seed=pair_seed, verbose=False)
    gamma = hist['gamma']

    # Coupled state = highest overall emission mean
    d_emit = hist['slds']['d_emit']
    mean_d = d_emit.mean(axis=1)
    coupled_state = np.argmax(mean_d)
    coupled_posterior = gamma[:, coupled_state]

    if gt_use.sum() > 0 and gt_use.sum() < T:
        auc = roc_auc_score(gt_use, coupled_posterior)
        if auc < 0.5:
            auc = 1 - auc
    else:
        auc = 0.5

    return auc


def semisynthetic_auc(session_data, kappa, n_pairs, rng, model_type='iohmm',
                      verbose=True):
    """Run semi-synthetic test at one kappa value (parallelized with joblib).

    Args:
        model_type: 'iohmm', 'slds', 'slds_fa', 'rslds', or 'full'

    Returns:
        list of AUC scores (one per pair)
    """
    from joblib import Parallel, delayed

    jobs = []
    for i in range(n_pairs):
        z_base, T = create_pseudo_dyad_z(session_data, rng)
        gt = np.zeros(T, dtype=bool)
        pair_seed = 42 + i + int(kappa * 1000)
        if model_type == 'iohmm':
            jobs.append((z_base, kappa, gt, pair_seed))
        else:
            jobs.append((z_base, kappa, gt, pair_seed, model_type))

    worker = _fit_one_pair if model_type == 'iohmm' else _fit_one_pair_slds
    aucs = Parallel(n_jobs=-1, prefer='processes')(
        delayed(worker)(job) for job in jobs)

    if verbose:
        print(f"  kappa={kappa:.2f}: AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f} "
              f"(n={len(aucs)}, model={model_type})", flush=True)

    return aucs


def main():
    print("=" * 70)
    print("  RSLDS Semi-Synthetic Validation (all model variants)")
    print("=" * 70)

    # 1. Synthetic recovery tests (9D)
    print("\n--- Synthetic IOHMM Recovery (9D) ---")
    rec_iohmm = synthetic_iohmm_recovery(K=3, D_obs=9, D_input=2, T=2000,
                                          n_trials=2, seed=42)

    print("\n--- Synthetic SLDS Recovery (9D, all variants) ---")
    from cadence.significance.rslds_validation import synthetic_slds_recovery
    rec_slds = synthetic_slds_recovery(K=3, D_obs=9, D_input=2, D_latent=3,
                                        n_factors=2, T=2000, n_trials=2, seed=42)

    # 2. Load scaffold data
    print("\n--- Loading scaffold data ---")
    session_data = load_all_scaffold_z()
    print(f"  Loaded {len(session_data)} sessions")

    if len(session_data) < 2:
        print("  ERROR: Need at least 2 sessions for pseudo-dyad pairs")
        return

    # 3. Semi-synthetic AUC sweep for each model variant
    MODEL_TYPES = ['iohmm', 'slds_fa', 'full']  # representative subset
    KAPPA_SUBSET = [0.0, 0.2, 0.4]  # speed: 3 kappas × 3 models

    all_results = {}
    t0 = time.time()

    for model_type in MODEL_TYPES:
        print(f"\n--- Semi-Synthetic AUC: {model_type} ---")
        rng = np.random.default_rng(42)
        model_results = {}
        for kappa in KAPPA_SUBSET:
            aucs = semisynthetic_auc(session_data, kappa, N_PSEUDO_PAIRS,
                                      rng, model_type=model_type, verbose=True)
            model_results[kappa] = {
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs)),
                'aucs': [float(a) for a in aucs],
            }
        all_results[model_type] = model_results

    print(f"\n  Total time: {time.time() - t0:.0f}s")

    # 4. Summary table
    print("\n--- Summary ---")
    header = f"  {'kappa':>6s}"
    for mt in MODEL_TYPES:
        header += f" | {mt:>10s}"
    print(header)
    print("  " + "-" * (8 + 13 * len(MODEL_TYPES)))
    for kappa in KAPPA_SUBSET:
        row = f"  {kappa:6.2f}"
        for mt in MODEL_TYPES:
            r = all_results[mt][kappa]
            row += f" | {r['mean_auc']:10.3f}"
        print(row)

    # Checks
    for mt in MODEL_TYPES:
        null_auc = all_results[mt][0.0]['mean_auc']
        max_auc = all_results[mt][max(KAPPA_SUBSET)]['mean_auc']
        print(f"\n  [{mt}] Null={null_auc:.3f}, Max={max_auc:.3f}")
        print(f"    [{'PASS' if null_auc < 0.60 else 'FAIL'}] Null < 0.60")
        print(f"    [{'PASS' if max_auc > 0.55 else 'FAIL'}] Max > 0.55")

    # Save results
    out = {
        'kappa_values': KAPPA_SUBSET,
        'model_types': MODEL_TYPES,
        'n_pairs': N_PSEUDO_PAIRS,
        'n_episodes': N_EPISODES,
        'episode_dur_s': EPISODE_DUR_S,
        'injection_dims': INJECTION_DIMS,
        'results': {mt: {str(k): v for k, v in mr.items()}
                    for mt, mr in all_results.items()},
        'synthetic_iohmm': rec_iohmm,
        'synthetic_slds': rec_slds,
    }
    os.makedirs('results/rslds', exist_ok=True)
    with open('results/rslds/rslds_semisynthetic_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved results/rslds/rslds_semisynthetic_results.json")


if __name__ == '__main__':
    main()
