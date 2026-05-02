"""CADENCE RSLDS V8 Full: IOHMM + SLDS (FA + recurrent) on 9D V8 features.

Runs the complete model hierarchy on V8 scaffold data:
  1. IOHMM baseline (K-sweep for model selection)
  2. SLDS with factor-analyzed emissions
  3. rSLDS with recurrent transitions
  4. Full model (FA + recurrent)

Compares all variants by BIC, ARI (if ground truth), and R² distribution.

Usage:
    python scripts/_run_rslds_v8_full.py
    python scripts/_run_rslds_v8_full.py --session y_06
    python scripts/_run_rslds_v8_full.py --K 4
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, IOHMMParams, iohmm_flexibility_metrics,
    build_observation_mask, fit_slds,
)
from scripts._run_rslds_phase2 import (
    load_scaffold, MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS,
    STATE_COLORS, FS_OUT,
)

# ── Constants ─────────────────────────────────────────────────────────

N_PCA_COMPONENTS = 2


# ── Fit one model variant ─────────────────────────────────────────────

def fit_variant(Y, U, obs_mask, K, variant, seed=42, max_iter=200, n_restarts=3):
    """Fit a single model variant.

    Args:
        variant: 'iohmm', 'slds_fa', 'rslds', 'full'

    Returns:
        dict with results
    """
    n_mod = Y.shape[1]
    t0 = time.time()

    if variant == 'iohmm':
        cfg = IOHMMConfig(K=K, D_obs=n_mod, D_input=U.shape[1],
                          n_restarts=n_restarts, max_em_iter=max_iter)
        model = IOHMM(cfg)
        params, history = model.fit(Y, U, obs_mask, seed=seed, verbose=True)
        gamma = history['gamma']
        viterbi = model.viterbi(Y, U, obs_mask, params)
        bic = history['bic']
        ll = history['final_ll']
        emission_mu = params.mu
        emission_sigma2 = params.sigma2

    else:
        D_latent = min(3, n_mod - 1)
        if variant == 'slds_fa':
            cfg = IOHMMConfig(K=K, D_obs=n_mod, D_input=U.shape[1],
                              D_latent=D_latent, n_factors=2,
                              n_restarts=n_restarts, max_em_iter=max_iter)
        elif variant == 'rslds':
            cfg = IOHMMConfig(K=K, D_obs=n_mod, D_input=U.shape[1],
                              D_latent=D_latent, recurrent=True,
                              n_restarts=n_restarts, max_em_iter=max_iter)
        elif variant == 'full':
            cfg = IOHMMConfig(K=K, D_obs=n_mod, D_input=U.shape[1],
                              D_latent=D_latent, n_factors=2, recurrent=True,
                              n_restarts=n_restarts, max_em_iter=max_iter)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        params, history = fit_slds(Y, U, obs_mask, cfg, seed=seed, verbose=True)
        gamma = history['gamma']
        # Viterbi from IOHMM on posteriors
        model = IOHMM(cfg)
        viterbi = np.argmax(gamma, axis=1)
        bic = history['bic']
        ll = history['final_ll']
        emission_mu = history['slds']['d_emit']
        emission_sigma2 = np.ones((K, n_mod))  # SLDS doesn't expose diagonal easily

    elapsed = time.time() - t0
    flex = iohmm_flexibility_metrics(gamma, fs=FS_OUT)

    # Between-state R² per modality
    grand_mean = Y.mean(axis=0)
    ss_total = np.sum((Y - grand_mean) ** 2, axis=0)
    ss_between = np.zeros(n_mod)
    for k in range(K):
        mask_k = viterbi == k
        if mask_k.sum() > 0:
            state_mean = Y[mask_k].mean(axis=0)
            ss_between += mask_k.sum() * (state_mean - grand_mean) ** 2
    r2_between = ss_between / np.maximum(ss_total, 1e-10)

    return {
        'variant': variant,
        'K': K,
        'bic': float(bic),
        'll': float(ll),
        'elapsed_s': elapsed,
        'flexibility': flex,
        'emission_mu': emission_mu.tolist(),
        'r2_between': {k: float(r2_between[i]) for i, k in enumerate(MODALITY_KEYS)},
        'gamma': gamma,
        'viterbi': viterbi,
    }


# ── Visualization ────────────────────────────────────────────────────

def plot_variant_comparison(t_common, Y, variant_results, session_name, out_path):
    """Multi-panel comparison of model variants."""
    n_variants = len(variant_results)
    n_mod = len(MODALITY_KEYS)

    fig, axes = plt.subplots(n_mod + n_variants, 1,
                             figsize=(26, 2.0 * (n_mod + n_variants)),
                             gridspec_kw={'height_ratios': [1]*n_mod + [0.6]*n_variants},
                             sharex=True)

    # z-timecourse panels
    for idx, (key, name, color) in enumerate(zip(MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS)):
        ax = axes[idx]
        ax.plot(t_common, Y[:, idx], color=color, linewidth=0.5, alpha=0.7)
        ax.axhline(0, color='gray', linewidth=0.3, alpha=0.4)
        ax.axhline(2, color='gray', linewidth=0.3, linestyle='--', alpha=0.3)
        ax.set_ylabel(f'{name}', fontsize=7)

    # Viterbi strips per variant
    for v_idx, vr in enumerate(variant_results):
        ax = axes[n_mod + v_idx]
        K = vr['K']
        for k in range(K):
            mask_k = vr['viterbi'] == k
            ax.fill_between(t_common, 0, 1, where=mask_k,
                            color=STATE_COLORS[k % len(STATE_COLORS)], alpha=0.7)
        ax.set_ylabel(f'{vr["variant"]}\nK={K}', fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        bic = vr['bic']
        ax.text(0.99, 0.8, f'BIC={bic:.0f}', transform=ax.transAxes,
                fontsize=7, ha='right', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('LSL time (s)', fontsize=9)

    # Title with R² comparison
    r2_lines = []
    for vr in variant_results:
        r2_str = ' '.join(f'{k[:6]}={v:.3f}' for k, v in vr['r2_between'].items())
        r2_lines.append(f"  {vr['variant']:10s} BIC={vr['bic']:.0f} R²: {r2_str}")

    fig.suptitle(f'{session_name} — V8 RSLDS Model Comparison\n' + '\n'.join(r2_lines),
                 fontsize=8, fontfamily='monospace', ha='left', x=0.02, va='top', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main pipeline ────────────────────────────────────────────────────

def run_session(session_name, K=4, max_iter=200, n_restarts=3):
    """Run all model variants on one session."""
    print(f"\n{'='*70}")
    print(f"  RSLDS V8 Full — {session_name}")
    print(f"{'='*70}")
    t_wall = time.time()

    # Load V8 scaffold
    z_traces, t_common, session_info = load_scaffold(session_name)
    if z_traces is None:
        return None
    T = len(t_common)
    print(f"  Loaded: {T} pts, {T/FS_OUT:.0f}s "
          f"({'V8' if session_info.get('_version') == 'v8' else 'V7 legacy'})")

    # Build observation matrix (V8 data is already prewhitened+standardized)
    Y = np.column_stack([z_traces[k] for k in MODALITY_KEYS]).astype(np.float64)
    U = np.zeros((T, N_PCA_COMPONENTS), dtype=np.float64)  # no slow covariates for V8
    obs_mask = build_observation_mask(z_traces, MODALITY_KEYS, T)

    # Fit all variants
    VARIANTS = ['iohmm', 'slds_fa', 'rslds', 'full']
    results = []

    for variant in VARIANTS:
        print(f"\n--- {variant} (K={K}) ---")
        r = fit_variant(Y, U, obs_mask, K, variant,
                        seed=42, max_iter=max_iter, n_restarts=n_restarts)
        results.append(r)
        print(f"  BIC={r['bic']:.0f}, LL={r['ll']:.0f}, {r['elapsed_s']:.0f}s")
        print(f"  Usage: {[f'{u:.1%}' for u in r['flexibility']['state_usage']]}")
        r2 = r['r2_between']
        top3 = sorted(r2.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top R²: {', '.join(f'{k}={v:.4f}' for k, v in top3)}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Variant':>10s} |    BIC   |    LL    | Trans | Entropy | Time")
    print(f"  " + "-" * 65)
    for r in results:
        print(f"  {r['variant']:>10s} | {r['bic']:8.0f} | {r['ll']:8.0f} | "
              f"{r['flexibility']['n_transitions']:5d} | "
              f"{r['flexibility']['shannon_entropy']:.3f}   | {r['elapsed_s']:.0f}s")

    print(f"\n  Between-state R² per modality:")
    print(f"  {'Variant':>10s} | " + ' '.join(f'{k:>10s}' for k in MODALITY_KEYS))
    print(f"  " + "-" * (12 + 11 * len(MODALITY_KEYS)))
    for r in results:
        vals = ' '.join(f'{r["r2_between"][k]:10.4f}' for k in MODALITY_KEYS)
        print(f"  {r['variant']:>10s} | {vals}")

    # Save
    out_dir = f'results/rslds/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    # Strip non-serializable arrays for JSON
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k not in ('gamma', 'viterbi')}
        json_results.append(jr)

    with open(os.path.join(out_dir, 'rslds_v8_full_results.json'), 'w') as f:
        json.dump({
            'session': session_name,
            'K': K,
            'modality_keys': MODALITY_KEYS,
            'variants': json_results,
        }, f, indent=2)

    # Save detailed NPZ
    save_dict = {'t_common': t_common, 'Y': Y}
    for r in results:
        prefix = r['variant']
        save_dict[f'{prefix}_gamma'] = r['gamma'].astype(np.float32)
        save_dict[f'{prefix}_viterbi'] = r['viterbi'].astype(np.int8)
    np.savez_compressed(os.path.join(out_dir, 'rslds_v8_full_results.npz'), **save_dict)

    # Visualization
    plot_variant_comparison(t_common, Y, results, session_name,
                            os.path.join(out_dir, 'rslds_v8_model_comparison.png'))

    total = time.time() - t_wall
    print(f"\n  Total: {total:.0f}s")
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RSLDS V8 Full model comparison')
    parser.add_argument('--session', nargs='+', default=['y_06'])
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--max-iter', type=int, default=200)
    args = parser.parse_args()

    for sname in args.session:
        run_session(sname, K=args.K, max_iter=args.max_iter, n_restarts=args.restarts)


if __name__ == '__main__':
    main()
