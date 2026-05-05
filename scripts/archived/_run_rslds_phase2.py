"""CADENCE RSLDS Phase 2: IOHMM coupling state dynamics model.

Fits an Input-Output HMM to 9D V8 z-timecourses (prewhitened + standardized)
with optional z_slow PCs as transition covariates.

V8 observation space (9D):
  EEG theta/alpha/beta, BL expression/state, ECG LF/HF, Resp, Pose

Pipeline:
  1. Load V8 scaffold z-timecourses (NPZ, already prewhitened+standardized)
  2. Optional spectral decomposition: z_slow (<0.01 Hz) / z_fast (residual)
  3. PCA on z_slow channels → 2D transition covariates
  4. Fit IOHMM with K-sweep (BIC) or fixed K
  5. Extract state posteriors, Viterbi path, flexibility metrics
  6. Save results + timeline visualization

Usage:
    python scripts/_run_rslds_phase2.py                    # y_06 only
    python scripts/_run_rslds_phase2.py --session y_06 y_17
    python scripts/_run_rslds_phase2.py --all              # all sessions
    python scripts/_run_rslds_phase2.py --k-sweep 2 5      # K selection sweep
    python scripts/_run_rslds_phase2.py --K 3              # fixed K
    python scripts/_run_rslds_phase2.py --no-decompose     # skip spectral decomposition
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA

from cadence.significance.rslds_model import (
    IOHMM, IOHMMConfig, IOHMMParams, iohmm_flexibility_metrics,
    build_observation_mask,
)

# ── Constants ─────────────────────────────────────────────────────────

FS_OUT = 2.0
SLOW_CUTOFF_HZ = 0.01
N_PCA_COMPONENTS = 2

MODALITY_KEYS = ['eeg_theta', 'eeg_alpha', 'eeg_beta',
                 'bl_expr', 'bl_state',
                 'ecg_lf', 'ecg_hf',
                 'resp', 'pose']
MODALITY_NAMES = ['EEG theta', 'EEG alpha', 'EEG beta',
                  'BL expression', 'BL state',
                  'ECG LF (SNS)', 'ECG HF (PNS)',
                  'Resp phase', 'Pose velocity']
MODALITY_COLORS = ['#90CAF9', '#A5D6A7', '#FFCC80',
                   '#E91E63', '#9C27B0',
                   '#FF5722', '#795548',
                   '#607D8B', '#4CAF50']

CONDITION_ORDER = ['base_EO', 'base_EC', 'conv_1', 'meditate_B', 'meditate_K', 'conv_2']
CONDITION_COLORS = {
    'base_EO': '#E3F2FD', 'base_EC': '#E8EAF6',
    'conv_1': '#FFF3E0', 'conv_2': '#FFF3E0',
    'meditate_B': '#F3E5F5', 'meditate_K': '#E8F5E9',
}

STATE_COLORS = ['#FF6F00', '#1565C0', '#2E7D32', '#6A1B9A', '#C62828']


# ── Spectral decomposition ───────────────────────────────────────────

def spectral_decompose(z_traces: dict, modality_keys: list,
                       fs: float = FS_OUT, cutoff: float = SLOW_CUTOFF_HZ,
                       n_pca: int = N_PCA_COMPONENTS):
    """Decompose z-timecourses into z_fast (coupling) + z_slow PCs (shared state).

    Returns:
        z_fast: dict of {key: (T,) array}
        z_slow_pcs: (T, n_pca) array
        decomp_info: dict with variance fractions, PCA loadings
    """
    sos = butter(4, cutoff, btype='low', fs=fs, output='sos')
    z_fast = {}
    z_slow_list = []

    for key in modality_keys:
        z = z_traces[key]
        if np.abs(z).max() < 1e-8:
            z_fast[key] = np.zeros_like(z)
            z_slow_list.append(np.zeros_like(z))
        else:
            z_s = sosfiltfilt(sos, z).astype(np.float32)
            z_fast[key] = (z - z_s).astype(np.float32)
            z_slow_list.append(z_s)

    z_slow_matrix = np.column_stack(z_slow_list)  # (T, D)
    active = [i for i in range(len(modality_keys))
              if np.abs(z_slow_matrix[:, i]).max() > 1e-8]

    if len(active) >= n_pca:
        pca = PCA(n_components=n_pca)
        pcs = pca.fit_transform(z_slow_matrix[:, active]).astype(np.float32)
        loadings_full = np.zeros((n_pca, len(modality_keys)), dtype=np.float32)
        for idx, col in enumerate(active):
            loadings_full[:, col] = pca.components_[:, idx]
        explained = pca.explained_variance_ratio_.tolist()
    else:
        pcs = np.zeros((len(z_slow_list[0]), n_pca), dtype=np.float32)
        loadings_full = np.zeros((n_pca, len(modality_keys)), dtype=np.float32)
        explained = [0.0] * n_pca

    # Variance fractions
    total_var = {k: float(np.var(z_traces[k])) for k in modality_keys}
    slow_frac = {}
    for i, k in enumerate(modality_keys):
        tv = total_var[k]
        sv = float(np.var(z_slow_list[i]))
        slow_frac[k] = sv / max(tv, 1e-10)

    decomp_info = {
        'slow_cutoff_hz': cutoff,
        'slow_fraction': slow_frac,
        'pca_explained_variance_ratio': explained,
        'pca_loadings': loadings_full.tolist(),
    }

    return z_fast, pcs, decomp_info


# ── Load scaffold data ───────────────────────────────────────────────

def load_scaffold(session_name: str):
    """Load V8 scaffold NPZ + JSON for a session.

    Tries V8 first, falls back to V7 (legacy) format.

    Returns:
        z_traces: dict of {key: (T,) array} — already prewhitened+standardized for V8
        t_common: (T,) timestamps
        session_info: dict from JSON
    """
    base = f'results/rslds/{session_name}'

    # Try V8 first
    npz_v8 = os.path.join(base, 'rslds_scaffold_v8_ztimecourses.npz')
    json_v8 = os.path.join(base, 'rslds_scaffold_v8_results.json')

    if os.path.exists(npz_v8):
        data = np.load(npz_v8)
        t_common = data['t_common']
        z_traces = {}
        for key in MODALITY_KEYS:
            z_key = f'z_{key}'
            if z_key in data:
                z_traces[key] = data[z_key]
            else:
                z_traces[key] = np.zeros(len(t_common), dtype=np.float32)

        session_info = {}
        if os.path.exists(json_v8):
            with open(json_v8) as f:
                session_info = json.load(f)
        session_info['_version'] = 'v8'
        return z_traces, t_common, session_info

    # Fallback to legacy V7
    npz_path = os.path.join(base, 'rslds_scaffold_ztimecourses.npz')
    json_path = os.path.join(base, 'rslds_scaffold_results.json')

    if not os.path.exists(npz_path):
        print(f"  WARNING: No scaffold data for {session_name}")
        return None, None, None

    data = np.load(npz_path)
    t_common = data['t_common']
    z_traces = {}
    # Map V7 keys to V8 keys
    v7_to_v8 = {
        'eeg': None,  # V7 had combined EEG, skip
        'eeg_theta': 'eeg_theta', 'eeg_alpha': 'eeg_alpha', 'eeg_beta': 'eeg_beta',
        'bl_expr': 'bl_expr', 'bl_state': 'bl_state',
        'ecg_sns': 'ecg_lf', 'ecg_pns': 'ecg_hf',
        'resp': 'resp', 'pose': 'pose',
    }
    for key in MODALITY_KEYS:
        z_key = f'z_{key}'
        if z_key in data:
            z_traces[key] = data[z_key]
        else:
            # Try V7 name mapping
            found = False
            for v7k, v8k in v7_to_v8.items():
                if v8k == key and f'z_{v7k}' in data:
                    z_traces[key] = data[f'z_{v7k}']
                    found = True
                    break
            if not found:
                z_traces[key] = np.zeros(len(t_common), dtype=np.float32)

    session_info = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            session_info = json.load(f)
    session_info['_version'] = 'v7_legacy'
    return z_traces, t_common, session_info


# ── Visualization ────────────────────────────────────────────────────

def plot_phase2_timeline(session_name, t_common, z_fast, z_slow_pcs,
                         gamma, viterbi_path, params, session_info,
                         out_path):
    """Multi-panel timeline: z_fast traces + state posteriors + Viterbi."""
    K = gamma.shape[1]
    n_mod = len(MODALITY_KEYS)

    # Panels: n_mod z_fast traces + 1 slow PCs + 1 state posteriors + 1 Viterbi strip
    n_panels = n_mod + 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(26, 2.8 * n_panels),
                             gridspec_kw={'height_ratios': [1]*n_mod + [0.8, 1, 0.5]},
                             sharex=True)

    # Condition shading
    segments = []
    for seg_name in CONDITION_ORDER:
        cf = session_info.get('condition_flexibility', {})
        if seg_name in cf:
            # Get segment boundaries from session grid
            cg = session_info.get('common_grid', {})
            pass  # we'll shade from condition_flexibility keys

    # Get condition boundaries from session_info if available
    cond_flex = session_info.get('condition_flexibility', {})

    # Plot condition shading on all axes (approximate from session grid)
    # We don't have exact boundaries in JSON, so skip condition shading if unavailable

    for ax in axes:
        ax.set_xlim(t_common[0], t_common[-1])

    # z_fast traces
    for idx, (key, name, color) in enumerate(zip(MODALITY_KEYS, MODALITY_NAMES, MODALITY_COLORS)):
        ax = axes[idx]
        z = z_fast.get(key, np.zeros(len(t_common)))
        ax.plot(t_common, z, color=color, linewidth=0.5, alpha=0.7)
        ax.axhline(0, color='gray', linewidth=0.3, alpha=0.4)
        ax.axhline(2.0, color='gray', linewidth=0.3, linestyle='--', alpha=0.3)

        # Color background by dominant state
        for k in range(K):
            state_mask = viterbi_path == k
            if state_mask.sum() == 0:
                continue
            sig_regions = np.diff(np.concatenate([[0], state_mask.astype(int), [0]]))
            starts = np.where(sig_regions == 1)[0]
            ends = np.where(sig_regions == -1)[0]
            for s, e in zip(starts, ends):
                if s < len(t_common) and e <= len(t_common):
                    ax.axvspan(t_common[max(0, s)], t_common[min(e-1, len(t_common)-1)],
                               alpha=0.08, color=STATE_COLORS[k % len(STATE_COLORS)])

        ax.set_ylabel(f'{name}\nz_fast', fontsize=8)
        cf = float((np.abs(z) > 2.0).mean())
        ax.text(0.01, 0.92, f'cf={cf:.1%}', transform=ax.transAxes, fontsize=7,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Slow PCs panel
    ax_pcs = axes[n_mod]
    for pc_idx in range(min(z_slow_pcs.shape[1], 2)):
        ax_pcs.plot(t_common, z_slow_pcs[:, pc_idx],
                    linewidth=1.0, alpha=0.8, label=f'PC{pc_idx+1}')
    ax_pcs.legend(fontsize=7, loc='upper right')
    ax_pcs.set_ylabel('z_slow\nPCs', fontsize=8)
    ax_pcs.axhline(0, color='gray', linewidth=0.3, alpha=0.4)

    # State posteriors panel
    ax_gamma = axes[n_mod + 1]
    for k in range(K):
        ax_gamma.fill_between(t_common, 0, gamma[:, k],
                              alpha=0.5, color=STATE_COLORS[k % len(STATE_COLORS)],
                              label=f'S{k}')
    ax_gamma.set_ylabel('State\nposterior', fontsize=8)
    ax_gamma.set_ylim(0, 1)
    ax_gamma.legend(fontsize=7, loc='upper right', ncol=K)

    # Viterbi strip
    ax_vit = axes[n_mod + 2]
    for k in range(K):
        mask_k = viterbi_path == k
        ax_vit.fill_between(t_common, 0, 1, where=mask_k,
                            color=STATE_COLORS[k % len(STATE_COLORS)], alpha=0.7)
    ax_vit.set_ylabel('Viterbi', fontsize=8)
    ax_vit.set_ylim(0, 1)
    ax_vit.set_yticks([])
    ax_vit.set_xlabel('LSL time (s)', fontsize=9)

    # Emission profile annotations
    title_parts = [f'{session_name} — IOHMM K={K}']
    for k in range(K):
        usage = gamma[:, k].mean()
        top_dims = np.argsort(params.mu[k])[::-1][:3]
        dim_str = ', '.join(f'{MODALITY_KEYS[d]}={params.mu[k,d]:.2f}' for d in top_dims)
        title_parts.append(f'  S{k} ({usage:.0%}): {dim_str}')
    fig.suptitle('\n'.join(title_parts), fontsize=10, fontfamily='monospace',
                 ha='left', x=0.02, va='top', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Single session pipeline ──────────────────────────────────────────

def run_session(session_name: str, K_range=None, K_fixed=None,
                n_restarts=5, max_iter=200, no_decompose=False):
    """Run Phase 2 IOHMM fitting for one session."""
    print(f"\n{'='*70}")
    print(f"  RSLDS Phase 2 — {session_name}")
    print(f"{'='*70}")
    t_wall = time.time()

    # 1. Load scaffold data
    z_traces, t_common, session_info = load_scaffold(session_name)
    if z_traces is None:
        return None
    T = len(t_common)
    is_v8 = session_info.get('_version', '') == 'v8'
    print(f"  Loaded: {T} timepoints, {T/FS_OUT:.0f}s (format: {'V8' if is_v8 else 'V7 legacy'})")

    # 2. Spectral decomposition (optional for V8 — data already prewhitened)
    if no_decompose or is_v8:
        print(f"\n  Skipping spectral decomposition (V8 data already prewhitened)")
        z_fast = z_traces
        z_slow_pcs = np.zeros((T, N_PCA_COMPONENTS), dtype=np.float32)
        decomp_info = {'slow_cutoff_hz': 0, 'slow_fraction': {k: 0.0 for k in MODALITY_KEYS},
                       'pca_explained_variance_ratio': [0.0] * N_PCA_COMPONENTS}
    else:
        print(f"\n  Spectral decomposition...", flush=True)
        z_fast, z_slow_pcs, decomp_info = spectral_decompose(z_traces, MODALITY_KEYS)
        for key, name in zip(MODALITY_KEYS, MODALITY_NAMES):
            sf = decomp_info['slow_fraction'][key]
            print(f"    {name:25s}: {sf:5.1%} slow")
        print(f"    PCA explained: {decomp_info['pca_explained_variance_ratio']}")

    # 3. Build observation matrix and mask
    Y = np.column_stack([z_fast[k] for k in MODALITY_KEYS]).astype(np.float64)
    U = z_slow_pcs.astype(np.float64)
    # For V8: obs_mask may be in NPZ, otherwise build from z_traces
    obs_mask = build_observation_mask(z_traces, MODALITY_KEYS, T)
    n_missing = (~obs_mask).any(axis=0).sum()
    if n_missing > 0:
        missing_mods = [MODALITY_KEYS[i] for i in range(len(MODALITY_KEYS))
                        if not obs_mask[:, i].all()]
        print(f"  Missing modalities: {missing_mods}")

    # 4. K-sweep or fixed K
    if K_range is not None:
        print(f"\n  K-sweep: {K_range[0]} to {K_range[1]}")
        bics = {}
        all_results = {}
        for K_test in range(K_range[0], K_range[1] + 1):
            print(f"\n  --- K={K_test} ---")
            cfg = IOHMMConfig(K=K_test, D_obs=len(MODALITY_KEYS),
                              D_input=U.shape[1],
                              n_restarts=n_restarts, max_em_iter=max_iter)
            model = IOHMM(cfg)
            params, history = model.fit(Y, U, obs_mask, seed=42, verbose=True)
            bics[K_test] = history['bic']
            all_results[K_test] = (params, history, model)

        best_K = min(bics, key=bics.get)
        print(f"\n  BIC scores: {', '.join(f'K={k}:{v:.0f}' for k, v in sorted(bics.items()))}")
        print(f"  Selected K={best_K}")

        params, history, model = all_results[best_K]
    else:
        K = K_fixed or 3
        print(f"\n  Fitting K={K}...")
        cfg = IOHMMConfig(K=K, D_obs=len(MODALITY_KEYS), D_input=U.shape[1],
                          n_restarts=n_restarts, max_em_iter=max_iter)
        model = IOHMM(cfg)
        params, history = model.fit(Y, U, obs_mask, seed=42, verbose=True)
        best_K = K

    # 5. Extract results
    gamma = history['gamma']
    viterbi_path = model.viterbi(Y, U, obs_mask, params)
    flex = iohmm_flexibility_metrics(gamma, fs=FS_OUT)

    print(f"\n  State usage: {[f'{u:.1%}' for u in flex['state_usage']]}")
    print(f"  Transitions: {flex['n_transitions']}, rate: {flex['transition_rate_hz']:.4f} Hz")
    print(f"  Entropy: {flex['shannon_entropy']:.3f}")

    # Per-state emission profiles
    print(f"\n  Emission profiles (z means):")
    print(f"  {'State':>6s} | " + ' '.join(f'{k:>10s}' for k in MODALITY_KEYS))
    print(f"  " + "-" * (10 + 11 * len(MODALITY_KEYS)))
    for k in range(best_K):
        vals = ' '.join(f'{params.mu[k, d]:+10.3f}' for d in range(len(MODALITY_KEYS)))
        print(f"  S{k} ({flex['state_usage'][k]:4.1%}) | {vals}")

    # 6. Save results
    out_dir = f'results/rslds/{session_name}'
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(os.path.join(out_dir, 'rslds_phase2_results.npz'),
        t_common=t_common,
        gamma=gamma.astype(np.float32),
        viterbi_path=viterbi_path.astype(np.int8),
        emission_mu=params.mu,
        emission_sigma2=params.sigma2,
        W_trans=params.W_trans,
        S_trans=params.S_trans,
        z_slow_pcs=z_slow_pcs,
        **{f'z_fast_{k}': z_fast[k] for k in MODALITY_KEYS})

    results = {
        'session': session_name,
        'K': best_K,
        'bic': float(history['bic']),
        'log_likelihood': float(history['final_ll']),
        'n_params': int(history['n_params']),
        'n_iters': int(history['n_iters']),
        'flexibility': flex,
        'emission_mu': params.mu.tolist(),
        'emission_sigma2': params.sigma2.tolist(),
        'spectral_decomposition': decomp_info,
        'params': params.to_dict(),
    }

    with open(os.path.join(out_dir, 'rslds_phase2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 7. Visualization
    plot_phase2_timeline(
        session_name, t_common, z_fast, z_slow_pcs,
        gamma, viterbi_path, params, session_info,
        os.path.join(out_dir, 'rslds_phase2_timeline.png'))

    total = time.time() - t_wall
    print(f"\n  {session_name} Phase 2 complete in {total:.0f}s")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RSLDS Phase 2: IOHMM fitting')
    parser.add_argument('--session', nargs='+', default=['y_06'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--k-sweep', nargs=2, type=int, metavar=('K_MIN', 'K_MAX'))
    parser.add_argument('--K', type=int, default=None, help='Fixed K')
    parser.add_argument('--restarts', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--no-decompose', action='store_true',
                        help='Skip spectral decomposition (V8 data already prewhitened)')
    args = parser.parse_args()

    if args.all:
        # Look for V8 first, then V7 scaffold files
        v8_npzs = sorted(glob.glob('results/rslds/*/rslds_scaffold_v8_ztimecourses.npz'))
        v7_npzs = sorted(glob.glob('results/rslds/*/rslds_scaffold_ztimecourses.npz'))
        all_npzs = list(set(v8_npzs + v7_npzs))
        sessions = sorted(set(os.path.basename(os.path.dirname(p)) for p in all_npzs))
    else:
        sessions = args.session

    K_range = tuple(args.k_sweep) if args.k_sweep else None
    print(f"Sessions: {sessions}")
    if K_range:
        print(f"K-sweep: {K_range[0]} to {K_range[1]}")
    elif args.K:
        print(f"Fixed K: {args.K}")

    if len(sessions) > 1 and K_range is None:
        from joblib import Parallel, delayed
        print(f"  Fitting {len(sessions)} sessions in parallel...", flush=True)
        results_list = Parallel(n_jobs=-1, prefer='processes')(
            delayed(run_session)(sname, K_range=K_range, K_fixed=args.K,
                                 n_restarts=args.restarts, max_iter=args.max_iter,
                                 no_decompose=args.no_decompose)
            for sname in sessions)
        all_results = {sname: r for sname, r in zip(sessions, results_list) if r is not None}
    else:
        all_results = {}
        for sname in sessions:
            result = run_session(sname, K_range=K_range, K_fixed=args.K,
                                 n_restarts=args.restarts, max_iter=args.max_iter,
                                 no_decompose=args.no_decompose)
            if result:
                all_results[sname] = result

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Session':>12s} | K  |  BIC    |  LL     | Trans | Entropy | Usage")
        print(f"  " + "-" * 75)
        for sname, r in sorted(all_results.items()):
            usage = ' '.join(f'{u:.0%}' for u in r['flexibility']['state_usage'])
            print(f"  {sname:>12s} | {r['K']} | {r['bic']:7.0f} | {r['log_likelihood']:7.0f} | "
                  f"{r['flexibility']['n_transitions']:5d} | {r['flexibility']['shannon_entropy']:7.3f} | {usage}")


if __name__ == '__main__':
    main()
