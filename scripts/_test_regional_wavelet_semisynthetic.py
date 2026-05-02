"""Per-region anatomical wavelet coherence — semi-synthetic validation battery.

Walks away from AFFECT_AUS entirely. For each anatomical region (mouth, eye,
brow, cheek, nose), inject coupling on a region-specific AU subset and verify
that ONLY the matching region's wavelet-coherence channel responds; other
regions stay near AUC=0.50.

The headline output is a (n_scenarios, n_regions) AUC matrix per kappa — the
"region-specificity matrix". Diagonal dominance indicates clean per-region
detection without cross-region leakage.

Reuses cross-session pseudo-dyad construction from cadence.synthetic_v82
(P1 from session A, P2 from session B → guarantees kappa=0 → AUC≈0.50).

Does NOT run the full V11 scaffold pipeline — operates only on raw blendshapes
plus the new bl_wavelet_regional module. This isolates the validation from
V11's known structural problems.

Usage:
    python scripts/_test_regional_wavelet_semisynthetic.py             # full battery (42 pairs, 7 scenarios, 6 kappas)
    python scripts/_test_regional_wavelet_semisynthetic.py --quick     # 10 pairs, 3 kappas (~minutes)
    python scripts/_test_regional_wavelet_semisynthetic.py --scenarios R_mouth_smile R_brow_furrow
"""

import argparse
import json
import os
import sys
import time

# Hoist torch BEFORE numpy in the entry script (Windows DLL workaround).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.config import load_config
from cadence.constants import AU_REGIONS
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import generate_coupling_gate
from cadence.synthetic_v82 import (
    build_v82_pseudo_dyad,
    compute_auc_within_session,
)
from cadence.significance.bl_wavelet_regional import (
    compute_regional_pca,
    compute_regional_wavelet_coherence,
    inject_bl_regional,
    REGIONAL_SCENARIOS,
)


FS_BL = 30.0
DEFAULT_KAPPAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
QUICK_KAPPAS = [0.0, 0.20, 0.40]
DEFAULT_DURATION_S = 300.0
DEFAULT_N_PAIRS = 42
DEFAULT_GATE_CFG = {
    'duty_cycle': 0.35,
    'event_range_s': (5, 20),
    'ramp_s': 2.0,
}
OUT_DIR = 'results/regional_wavelet_semisynthetic'


# =========================================================================
# Session loading + pair construction
# =========================================================================

def load_available_sessions(config, max_sessions=None):
    """Load cached sessions that have BOTH participants' blendshape data."""
    cached = discover_cached_sessions(config['session_cache'])
    sessions = []
    for name, path in cached:
        try:
            c = load_session_from_cache(path, config)
            if 'p1_blendshapes' in c and 'p2_blendshapes' in c:
                sessions.append((name, c))
                if max_sessions and len(sessions) >= max_sessions:
                    break
        except Exception:
            continue
    return sessions


def generate_pseudo_dyad_pairs(sessions, n_pairs, seed=42):
    """Random cross-session pairs (P1 from session i, P2 from session j, i!=j)."""
    rng = np.random.default_rng(seed)
    n = len(sessions)
    if n < 2:
        return []
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        pairs.append((int(i), int(j)))
    return pairs


# =========================================================================
# Single-pair evaluation (one scenario × one kappa)
# =========================================================================

def evaluate_pair(cached_a, cached_b, scenario_name, kappa, seed,
                   duration_s=DEFAULT_DURATION_S, gate_cfg=None,
                   n_surrogates=200, regions=None, device='auto',
                   collect_pca=False, pooling='au_pooled'):
    """Inject region-specific coupling and measure regional AUCs.

    Returns dict:
        per_region_auc: {region_name: AUC of expression-band z vs gate}
        per_region_band_z: {region_name: {state, expression, speech} mean z}
        loadings (optional): {region_name: PC loadings dict from compute_regional_pca}
        gate_density: float (mean gate value, sanity check)
    Returns None if pair unusable.
    """
    if regions is None:
        regions = AU_REGIONS
    if gate_cfg is None:
        gate_cfg = DEFAULT_GATE_CFG

    scen = REGIONAL_SCENARIOS[scenario_name]

    # Resolve overlapping LSL window
    ts_a = cached_a.get('p1_blendshapes_ts')
    ts_b = cached_b.get('p2_blendshapes_ts')
    if ts_a is None or ts_b is None or len(ts_a) < 2 or len(ts_b) < 2:
        return None

    t_start = max(ts_a[0], ts_b[0])
    t_end = min(ts_a[-1], ts_b[-1])
    available = t_end - t_start
    if available < 60:
        return None
    dur = min(duration_s, available)
    t_end = t_start + dur

    base = build_v82_pseudo_dyad(cached_a, cached_b, t_start, t_end)
    if 'p1_blendshapes' not in base or 'p2_blendshapes' not in base:
        return None

    p1_bl = base['p1_blendshapes']
    p2_bl = base['p2_blendshapes']

    # Use same number of timepoints / channels for both
    n_t = min(len(p1_bl), len(p2_bl))
    n_ch = min(p1_bl.shape[1], p2_bl.shape[1])
    p1_bl = p1_bl[:n_t, :n_ch]
    p2_bl = p2_bl[:n_t, :n_ch]
    if n_t < int(60 * FS_BL):
        return None

    # 30 Hz gate
    gate_bl = generate_coupling_gate(n_t, FS_BL, gate_cfg, seed=seed)

    # Inject (skip when kappa is 0 → null pair, but generate "fake" gate for AUC
    # null integrity check)
    if kappa > 0:
        p1_inj, p2_inj = inject_bl_regional(
            p1_bl, p2_bl, kappa=kappa,
            au_subset=scen['au_subset'], lag_s=scen['lag_s'],
            gate=gate_bl, seed=seed, fs=FS_BL,
            secondary_aus=scen.get('secondary_aus'),
        )
    else:
        p1_inj = p1_bl.astype(np.float32, copy=True)
        p2_inj = p2_bl.astype(np.float32, copy=True)

    # Per-region wavelet coherence with surrogate z
    coh = compute_regional_wavelet_coherence(
        p1_inj, p2_inj, regions=regions, fs=FS_BL,
        n_surrogates=n_surrogates, seed=seed, device=device,
        pooling=pooling,
    )

    # AUC per region (gate ON vs OFF on expression-band z timecourse)
    per_region_auc = {}
    per_region_band_z = {}
    for region_name in regions.keys():
        if region_name not in coh:
            per_region_auc[region_name] = float('nan')
            per_region_band_z[region_name] = {}
            continue
        z_ts = coh[region_name]['band_z_ts']['expression']
        # AUC: gate-ON vs gate-OFF discrimination at 30 Hz
        per_region_auc[region_name] = compute_auc_within_session(z_ts, gate_bl)
        per_region_band_z[region_name] = {
            'state':      coh[region_name]['band_z_mean']['state'],
            'expression': coh[region_name]['band_z_mean']['expression'],
            'speech':     coh[region_name]['band_z_mean']['speech'],
        }

    out = {
        'per_region_auc': per_region_auc,
        'per_region_band_z': per_region_band_z,
        'gate_density': float(gate_bl.mean()),
    }

    if collect_pca:
        # PCA on the (uninjected) raw P1+P2 stacked panel — captures natural
        # regional structure rather than the injection signal.
        stacked = np.concatenate([p1_bl, p2_bl], axis=0)
        pca = compute_regional_pca(stacked, regions=regions, n_components=2)
        out['loadings'] = {
            r: {
                'loadings': info['loadings'].tolist(),
                'var_explained': info['var_explained'].tolist(),
                'au_indices': info['au_indices'],
            }
            for r, info in pca.items()
        }

    return out


# =========================================================================
# Batched runner (one scenario × all kappas × all pairs)
# =========================================================================

def run_scenario(scenario_name, sessions, pairs, kappas, duration_s,
                  n_surrogates, regions, device, n_jobs, collect_pca=False,
                  seed_base=42, pooling='au_pooled'):
    """Run all (pair × kappa) for a single scenario. Returns nested dict."""
    print(f"\n  Scenario: {scenario_name}  "
          f"target={REGIONAL_SCENARIOS[scenario_name]['target_region']}  "
          f"au_subset={REGIONAL_SCENARIOS[scenario_name]['au_subset']}")

    scenario_results = {}
    for kappa in kappas:
        t0 = time.time()
        print(f"    kappa={kappa:.2f}: ", end='', flush=True)

        def _one_pair(pi, pair):
            i_a, i_b = pair
            _, cached_a = sessions[i_a]
            _, cached_b = sessions[i_b]
            seed = seed_base + pi + int(kappa * 1000)
            return evaluate_pair(
                cached_a, cached_b, scenario_name, kappa, seed,
                duration_s=duration_s, n_surrogates=n_surrogates,
                regions=regions, device=device,
                collect_pca=collect_pca and pi < 5,  # only first few pairs for PCA
                pooling=pooling,
            )

        # Threads: GPU work releases the GIL via torch CUDA calls, and joblib's
        # threading backend avoids the Windows torch+loky fork issue noted in
        # the project memory.
        per_pair = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_one_pair)(pi, p) for pi, p in enumerate(pairs)
        )
        valid = [r for r in per_pair if r is not None]

        # Aggregate AUCs across pairs
        region_aucs = {r: [] for r in regions.keys()}
        region_band_z = {r: {'state': [], 'expression': [], 'speech': []}
                         for r in regions.keys()}
        for r in valid:
            for region_name in regions.keys():
                v = r['per_region_auc'].get(region_name, float('nan'))
                if not np.isnan(v):
                    region_aucs[region_name].append(v)
                bz = r['per_region_band_z'].get(region_name, {})
                for b in ('state', 'expression', 'speech'):
                    if b in bz and not np.isnan(bz[b]):
                        region_band_z[region_name][b].append(bz[b])

        kappa_summary = {}
        for region_name in regions.keys():
            vals = region_aucs[region_name]
            kappa_summary[region_name] = {
                'auc_mean': float(np.mean(vals)) if vals else 0.5,
                'auc_std':  float(np.std(vals)) if vals else 0.0,
                'n':        len(vals),
                'band_z_mean': {
                    b: (float(np.mean(region_band_z[region_name][b]))
                        if region_band_z[region_name][b] else 0.0)
                    for b in ('state', 'expression', 'speech')
                },
            }

        scenario_results[f'kappa_{kappa:.2f}'] = kappa_summary

        # Print top regions for this kappa
        ranked = sorted(kappa_summary.items(),
                        key=lambda x: -x[1]['auc_mean'])
        top_str = '  '.join(f"{r}={v['auc_mean']:.3f}" for r, v in ranked[:3])
        elapsed = time.time() - t0
        print(f"{len(valid)} pairs  {elapsed:.1f}s  top: {top_str}")

        # Optionally collect PCA loadings (first kappa, first few pairs only)
        if collect_pca and kappa == kappas[0] and any('loadings' in r for r in valid):
            scenario_results['loadings_sample'] = next(
                (r['loadings'] for r in valid if 'loadings' in r), None)

    return scenario_results


# =========================================================================
# Verification helpers
# =========================================================================

def evaluate_specificity_matrix(results, kappa_str='kappa_0.30'):
    """Build a (scenario × region) AUC matrix at a specific kappa.

    Returns:
        scenarios: list of scenario names
        regions:   list of region names
        matrix:    (n_scen, n_reg) ndarray of AUC means
    """
    scenarios = list(results.keys())
    regions_set = set()
    for s in scenarios:
        if kappa_str in results[s]:
            regions_set.update(results[s][kappa_str].keys())
    regions_list = sorted(regions_set)

    matrix = np.full((len(scenarios), len(regions_list)), np.nan, dtype=float)
    for i, s in enumerate(scenarios):
        if kappa_str not in results[s]:
            continue
        for j, r in enumerate(regions_list):
            entry = results[s][kappa_str].get(r)
            if entry:
                matrix[i, j] = entry['auc_mean']
    return scenarios, regions_list, matrix


def verify_acceptance(results, kappas):
    """Apply plan-defined acceptance criteria and print PASS/FAIL summary."""
    print('\n' + '=' * 72)
    print('ACCEPTANCE VERIFICATION')
    print('=' * 72)

    # 1. Null integrity at kappa=0
    if 0.0 in kappas:
        print('\n[V1] Null integrity at kappa=0 (target <= 0.55):')
        bad = []
        for scen, scen_data in results.items():
            k0 = scen_data.get('kappa_0.00')
            if not k0:
                continue
            for region, info in k0.items():
                if isinstance(info, dict) and 'auc_mean' in info:
                    if info['auc_mean'] > 0.55 or info['auc_mean'] < 0.45:
                        bad.append((scen, region, info['auc_mean']))
        if not bad:
            print('  PASS — all (scenario, region) AUCs at kappa=0 within [0.45, 0.55]')
        else:
            for s, r, v in bad[:10]:
                print(f'  WARN  {s} / {r}  AUC={v:.3f}')

    # 2. Diagonal dominance at the largest non-zero kappa available
    nonzero = [k for k in kappas if k > 0]
    if not nonzero:
        kref = None
    else:
        kref = max(nonzero)
    kref_str = f'kappa_{kref:.2f}' if kref is not None else None
    print(f'\n[V2] Region-specificity matrix at {kref_str}:')
    scen_list, reg_list, mat = (
        evaluate_specificity_matrix(results, kref_str) if kref_str
        else ([], [], np.zeros((0, 0)))
    )
    if mat.size == 0 or np.all(np.isnan(mat)):
        print(f'  SKIP - no {kref_str} results' if kref_str else '  SKIP - no nonzero kappa')
    else:
        # For each scenario, find target region(s) and check whether AUC >= 0.65
        scen_target_regions = {
            'R_mouth_smile':  ['mouth'],
            'R_mouth_frown':  ['mouth'],
            'R_brow_furrow':  ['brow'],
            'R_eye_squint':   ['eye'],
            'R_cheek_squint': ['cheek'],
            'R_nose_sneer':   ['nose'],
            'R_duchenne':     ['mouth', 'cheek'],  # cross-region positive control
        }
        ok = True
        for i, scen in enumerate(scen_list):
            targets = scen_target_regions.get(scen, [])
            non_targets = [r for r in reg_list if r not in targets]
            tgt_aucs = [mat[i, reg_list.index(r)] for r in targets if r in reg_list]
            other_aucs = [mat[i, reg_list.index(r)] for r in non_targets if r in reg_list]
            tgt_min = min(tgt_aucs) if tgt_aucs else 0.5
            other_max = max(other_aucs) if other_aucs else 0.5
            tag = 'OK ' if (tgt_min >= 0.65 and other_max <= 0.55) else 'WARN'
            if tag == 'WARN':
                ok = False
            print(f'  {tag}  {scen:18s}  target_min={tgt_min:.3f}  other_max={other_max:.3f}')
        print('  ' + ('PASS' if ok else 'FAIL') + ' overall diagonal dominance')

    # 3. Dose-response monotonicity per scenario per target region
    print('\n[V3] Dose-response monotonicity (target region, kappa increasing):')
    scen_target_regions = {
        'R_mouth_smile': 'mouth',  'R_mouth_frown': 'mouth',
        'R_brow_furrow': 'brow',   'R_eye_squint': 'eye',
        'R_cheek_squint': 'cheek', 'R_nose_sneer': 'nose',
        'R_duchenne': 'mouth',
    }
    for scen, region in scen_target_regions.items():
        if scen not in results:
            continue
        seq = []
        for k in kappas:
            ks = f'kappa_{k:.2f}'
            entry = results[scen].get(ks, {}).get(region)
            if entry:
                seq.append((k, entry['auc_mean']))
        if len(seq) < 2:
            continue
        # Spearman-style monotonicity (counts inversions)
        vals = [v for _, v in seq]
        inversions = sum(1 for a, b in zip(vals, vals[1:]) if b < a - 0.02)
        tag = 'OK ' if inversions == 0 else 'WARN'
        seq_str = '  '.join(f'k={k:.2f}:{v:.3f}' for k, v in seq)
        print(f'  {tag}  {scen:18s} {region:6s}  {seq_str}')

    print('=' * 72 + '\n')


# =========================================================================
# CLI
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true',
                    help='Fast smoke run (10 pairs, 3 kappas)')
    ap.add_argument('--scenarios', nargs='+', default=None,
                    help='Subset of REGIONAL_SCENARIOS to run')
    ap.add_argument('--n-pairs', type=int, default=None,
                    help=f'Override number of pseudo-dyad pairs (default {DEFAULT_N_PAIRS}, --quick→10)')
    ap.add_argument('--n-surrogates', type=int, default=200,
                    help='Number of circular-shift surrogates per pair')
    ap.add_argument('--duration', type=float, default=DEFAULT_DURATION_S,
                    help='Window duration per pair in seconds')
    ap.add_argument('--n-jobs', type=int, default=4,
                    help='Joblib threads (GPU work releases GIL).')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--collect-pca', action='store_true',
                    help='Also save sample PC1/PC2 loadings per region (first 5 pairs).')
    ap.add_argument('--pooling', default='au_pooled',
                    choices=['au_pooled', 'pc1'],
                    help="Region-pooling method: 'au_pooled' sums cross-/auto-spectra "
                         "across AUs in region; 'pc1' fits PC1 per region per "
                         "participant and runs univariate coherence on the PC1 "
                         "timeseries (concentrates signal but loses some omnibus "
                         "info).")
    args = ap.parse_args()

    config = load_config(args.config)
    # Subdirectory per pooling method so AU-pooled vs PC1 results don't collide.
    args.out_dir = os.path.join(args.out_dir, args.pooling)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load sessions
    print('[1/3] Loading cached sessions...')
    sessions = load_available_sessions(config)
    print(f'  loaded {len(sessions)} sessions with BL data')
    if len(sessions) < 2:
        print('  Need at least 2 sessions with BL data; aborting.')
        return 1

    # Pairs
    n_pairs = args.n_pairs or (10 if args.quick else DEFAULT_N_PAIRS)
    pairs = generate_pseudo_dyad_pairs(sessions, n_pairs=n_pairs)
    print(f'  generated {len(pairs)} pseudo-dyad pairs')

    # Kappas + scenarios
    kappas = QUICK_KAPPAS if args.quick else DEFAULT_KAPPAS
    scenarios = args.scenarios or list(REGIONAL_SCENARIOS.keys())
    invalid = [s for s in scenarios if s not in REGIONAL_SCENARIOS]
    if invalid:
        print(f'  Unknown scenarios: {invalid}')
        return 1

    print(f'  kappas: {kappas}')
    print(f'  scenarios: {scenarios}')
    print(f'  regions: {list(AU_REGIONS.keys())}')
    print(f'  pooling: {args.pooling}')

    print('\n[2/3] Running battery...')
    t_battery = time.time()
    results = {}
    for scen in scenarios:
        results[scen] = run_scenario(
            scen, sessions, pairs, kappas,
            duration_s=args.duration,
            n_surrogates=args.n_surrogates,
            regions=AU_REGIONS,
            device=args.device,
            n_jobs=args.n_jobs,
            collect_pca=args.collect_pca,
            pooling=args.pooling,
        )
    elapsed = time.time() - t_battery
    print(f'\n  battery elapsed: {elapsed/60:.1f} min')

    # Save
    out_path = os.path.join(args.out_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'meta': {
                'n_sessions': len(sessions),
                'n_pairs': len(pairs),
                'kappas': list(kappas),
                'scenarios': scenarios,
                'regions': {k: list(v) for k, v in AU_REGIONS.items()},
                'duration_s': args.duration,
                'n_surrogates': args.n_surrogates,
                'device': args.device,
                'pooling': args.pooling,
                'elapsed_s': elapsed,
            },
            'results': results,
        }, f, indent=2)
    print(f'  saved {out_path}')

    print('\n[3/3] Acceptance verification...')
    verify_acceptance(results, kappas)

    return 0


if __name__ == '__main__':
    sys.exit(main())
