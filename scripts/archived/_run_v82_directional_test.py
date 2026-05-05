"""V8.2 Directional Test: asymmetry features + D_latent BIC sweep.

1. Scaffold all sessions with 18D features (15D + 3 asymmetry)
2. Test asymmetry condition separation (conversation vs baseline vs meditation/PE)
3. Ground truth: meditation/PE should show therapist-leading asymmetry
4. BIC sweep: D_latent = 2, 3, 4 on hierarchical rSLDS

Usage:
    python scripts/_run_v82_directional_test.py
"""

import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
from cadence.significance.rslds_model import IOHMMConfig, fit_hierarchical_slds, slds_m_step_emissions
from cadence.config import load_config

MODALITY_KEYS = [
    'imcoh_theta', 'imcoh_alpha', 'imcoh_beta',
    'conc_theta', 'conc_alpha', 'conc_beta',
    'dyn_theta', 'dyn_alpha', 'dyn_beta',
    'asym_theta', 'asym_alpha', 'asym_beta',
    'bl_expr', 'bl_activity_conc',
    'ecg_lf', 'ecg_hf', 'resp', 'pose',
]
FS_OUT = 2.0


def main():
    print("=" * 80)
    print("  V8.2 Directional Test: Asymmetry + D_latent BIC Sweep")
    print("=" * 80)

    # ── Step 1: Scaffold all sessions ────────────────────────────────
    print("\n[1/4] Scaffolding all sessions with 18D features...")
    config = load_config()
    os.system('source activate MCCT && python scripts/_run_scaffold_v82.py --all > /dev/null 2>&1')

    # Verify scaffolds
    npzs = sorted(glob.glob('results/rslds/*/scaffold_v82_ztimecourses.npz'))
    print(f"  {len(npzs)} sessions scaffolded")

    # Check dimensionality
    sample = np.load(npzs[0])
    available_keys = [k for k in MODALITY_KEYS if f'z_{k}' in sample]
    print(f"  Available features: {len(available_keys)}/{len(MODALITY_KEYS)}")
    if len(available_keys) < 18:
        print(f"  Missing: {[k for k in MODALITY_KEYS if f'z_{k}' not in sample]}")

    # ── Step 2: Condition separation test for asymmetry ──────────────
    print("\n[2/4] Testing asymmetry condition separation...")

    features_to_test = ['asym_theta', 'asym_alpha', 'asym_beta',
                        'conc_theta', 'conc_alpha', 'conc_beta',
                        'imcoh_alpha', 'imcoh_beta']

    # Collect per-session per-condition means
    all_cond_data = {}
    for npz_path in npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path)
        t = data['t_common']
        json_path = f'results/rslds/{name}/scaffold_v82_results.json'
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            info = json.load(f)
        segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

        session_conds = {}
        for seg_name, t0, t1 in segments:
            mask = (t >= t0) & (t <= t1)
            if mask.sum() < 10:
                continue
            # Normalize condition type
            if seg_name.startswith('base') or seg_name == 'baseline':
                ct = 'baseline'
            elif seg_name.startswith('conv'):
                ct = 'conversation'
            elif seg_name.startswith('PE'):
                ct = 'PE'
            elif seg_name.startswith('meditate'):
                ct = 'meditation'
            else:
                continue

            if ct not in session_conds:
                session_conds[ct] = {}
            for f in features_to_test:
                zk = f'z_{f}'
                if zk in data:
                    if f not in session_conds[ct]:
                        session_conds[ct][f] = []
                    session_conds[ct][f].append(float(data[zk][mask].mean()))

        # Average across segments of same type
        for ct in session_conds:
            for f in session_conds[ct]:
                session_conds[ct][f] = np.mean(session_conds[ct][f])
        all_cond_data[name] = session_conds

    # Wilcoxon tests: all pairwise condition comparisons for asymmetry
    pairs = [('conversation', 'baseline'), ('conversation', 'meditation'),
             ('conversation', 'PE'), ('meditation', 'baseline'), ('PE', 'baseline')]

    print(f"\n  ASYMMETRY CONDITION SEPARATION:")
    print(f"  {'Feature':>15s} | {'Comparison':>25s} | diff      | p-value | sig  | n")
    print(f"  " + "-" * 85)

    for f in features_to_test:
        for c1, c2 in pairs:
            v1 = []
            v2 = []
            for name, sc in all_cond_data.items():
                if c1 in sc and c2 in sc and f in sc.get(c1, {}) and f in sc.get(c2, {}):
                    v1.append(sc[c1][f])
                    v2.append(sc[c2][f])
            if len(v1) >= 4:
                diff = np.mean(v1) - np.mean(v2)
                try:
                    _, p = wilcoxon(v1, v2)
                except ValueError:
                    p = 1.0
                sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                comp = f'{c1[:4]} vs {c2[:4]}'
                print(f"  {f:>15s} | {comp:>25s} | {diff:+9.4f} | {p:.4f}  | {sig:4s} | {len(v1)}")

    # ── Step 3: Ground truth — meditation/PE asymmetry ───────────────
    print(f"\n[3/4] Ground truth: therapist-leading in meditation/PE...")
    print(f"  (P1=patient, P2=therapist in most sessions)")
    print(f"  During meditation/PE: therapist speaks (low alpha), patient silent (high alpha)")
    print(f"  Expected: asym_alpha = z_P1 - z_P2 should be POSITIVE (patient has more alpha)")

    for ct in ['meditation', 'PE']:
        vals = {}
        for name, sc in all_cond_data.items():
            if ct in sc:
                for f in ['asym_theta', 'asym_alpha', 'asym_beta']:
                    if f in sc[ct]:
                        if f not in vals:
                            vals[f] = []
                        vals[f].append(sc[ct][f])

        if vals:
            print(f"\n  {ct.upper()} (n={len(vals.get('asym_alpha', []))}):")
            for f in ['asym_theta', 'asym_alpha', 'asym_beta']:
                if f in vals and vals[f]:
                    v = vals[f]
                    direction = "P1>P2 (patient higher)" if np.mean(v) > 0 else "P2>P1 (therapist higher)"
                    print(f"    {f}: mean={np.mean(v):+.3f} +/- {np.std(v):.3f} — {direction}")

    # Conversation should be symmetric (near zero)
    conv_vals = {}
    for name, sc in all_cond_data.items():
        if 'conversation' in sc:
            for f in ['asym_theta', 'asym_alpha', 'asym_beta']:
                if f in sc['conversation']:
                    if f not in conv_vals:
                        conv_vals[f] = []
                    conv_vals[f].append(sc['conversation'][f])

    print(f"\n  CONVERSATION (n={len(conv_vals.get('asym_alpha', []))}):")
    for f in ['asym_theta', 'asym_alpha', 'asym_beta']:
        if f in conv_vals and conv_vals[f]:
            v = conv_vals[f]
            print(f"    {f}: mean={np.mean(v):+.3f} +/- {np.std(v):.3f}")

    # ── Step 4: D_latent BIC sweep ───────────────────────────────────
    print(f"\n[4/4] D_latent BIC sweep (hierarchical rSLDS)...")

    # Load all sessions
    sessions = []
    for npz_path in npzs:
        name = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path)
        t = data['t_common']
        Y = np.column_stack([data[f'z_{k}'] for k in available_keys]).astype(np.float64)
        obs_mask = data['obs_mask'] if 'obs_mask' in data else np.ones_like(Y, dtype=bool)
        # Trim obs_mask to match available features
        if obs_mask.shape[1] > Y.shape[1]:
            obs_mask = obs_mask[:, :Y.shape[1]]
        elif obs_mask.shape[1] < Y.shape[1]:
            obs_mask = np.hstack([obs_mask, np.ones((len(Y), Y.shape[1] - obs_mask.shape[1]), dtype=bool)])
        T, D = Y.shape
        U = np.zeros((T, 2), dtype=np.float64)
        sessions.append((Y, U, obs_mask, name))

    session_tuples = [(Y, U, mask) for Y, U, mask, name in sessions]
    D_obs = sessions[0][0].shape[1]

    print(f"  {len(sessions)} sessions, D_obs={D_obs}")

    results = {}
    for d_lat in [2, 3, 4]:
        print(f"\n  --- D_latent={d_lat} ---")
        t0 = time.time()
        cfg = IOHMMConfig(
            K=4, D_obs=D_obs, D_input=2, D_latent=d_lat,
            n_factors=2, recurrent=True,
            n_restarts=2, max_em_iter=80,
            sticky_strength=3.0,
            null_state=True, null_sigma2_cap=5.0,
        )
        try:
            result = fit_hierarchical_slds(session_tuples, cfg, seed=42, verbose=True)
            bic = result['bic']
            ll = result['ll_trace'][-1] if result['ll_trace'] else 0
            elapsed = time.time() - t0
            results[d_lat] = {'bic': bic, 'll': ll, 'elapsed': elapsed}
            print(f"  BIC={bic:.0f}, LL={ll:.0f}, {elapsed:.0f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[d_lat] = {'bic': float('inf'), 'll': 0, 'error': str(e)}

    # Summary
    print(f"\n{'='*80}")
    print(f"  D_LATENT BIC COMPARISON ({D_obs}D observations)")
    print(f"{'='*80}")
    print(f"  D_latent |      BIC      |      LL      | Time")
    print(f"  " + "-" * 50)
    for d_lat in sorted(results):
        r = results[d_lat]
        print(f"  {d_lat:>7d} | {r['bic']:13.0f} | {r.get('ll', 0):12.0f} | {r.get('elapsed', 0):.0f}s")

    best = min(results, key=lambda d: results[d]['bic'])
    print(f"\n  Best D_latent: {best} (BIC={results[best]['bic']:.0f})")

    # Save
    with open('results/rslds/v82_directional_test_results.json', 'w') as f:
        json.dump({
            'n_sessions': len(sessions),
            'D_obs': D_obs,
            'available_keys': available_keys,
            'd_latent_sweep': {str(k): v for k, v in results.items()},
            'best_d_latent': best,
        }, f, indent=2, default=str)
    print(f"\n  Saved results/rslds/v82_directional_test_results.json")


if __name__ == '__main__':
    main()
