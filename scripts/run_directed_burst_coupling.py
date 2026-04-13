"""V11 Post-Hoc: Directed EEG Burst Coupling (ECA + Transfer Entropy).

Who leads? Directed ECA scans lags 0-2s for therapist-leads vs patient-leads.
Transfer entropy asks: does P1's burst history reduce uncertainty about P2's
next burst? Both surrogate-calibrated via circular-shift.

Usage:
    python scripts/run_directed_burst_coupling.py                # y_06 only
    python scripts/run_directed_burst_coupling.py --session y_17
    python scripts/run_directed_burst_coupling.py --all
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.significance.directed_burst_coupling import eeg_directed_burst_coupling
from cadence.significance.fast_cycles import EEG_BANDS
from scripts.run_session_v6 import load_xdf_session

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
SCAFFOLD_DIR = 'results/v11'
OUT_DIR = 'results/v11/directed_burst_coupling'
HIERARCHICAL_RESULTS_PATH = 'results/v11/hierarchical/v11_hierarchical_results.json'
FS_OUT = 2.0

CONDS = ['base_EO', 'base_EC', 'conv_1', 'conv_2',
         'meditate_B', 'meditate_K', 'PE_1', 'PE_2']


def _load_state_labels(path=HIERARCHICAL_RESULTS_PATH):
    """Load index-ordered state labels from the V11 hierarchical fit.

    The hierarchical fit assigns labels by emission structure (COUP=argmax
    imcoh sum, SHARED=argmax conc sum), so the index→label permutation
    differs across refits. Always read it from the fit, not a hardcoded list.
    """
    with open(path) as f:
        labels = list(json.load(f)['state_labels'])
    assert sorted(labels) == ['COUP', 'NULL', 'OTHER', 'SHARED'], \
        f'Unexpected state labels in {path}: {labels}'
    return labels


def analyze_session(session_name, config, state_labels):
    """Compute directed ECA + TE for one session.

    Args:
        state_labels: index-ordered labels from the hierarchical fit.
            state_labels[si] is the label of rSLDS state with index si in
            results/v11/{session}/v11_rslds_results.npz['path'].
    """

    scaffold_npz = f'{SCAFFOLD_DIR}/{session_name}/scaffold_v11_ztimecourses.npz'
    scaffold_json = f'{SCAFFOLD_DIR}/{session_name}/scaffold_v11_results.json'
    rslds_npz = f'{SCAFFOLD_DIR}/{session_name}/v11_rslds_results.npz'

    if not os.path.exists(scaffold_npz):
        print(f"  No scaffold for {session_name}")
        return None

    data = np.load(scaffold_npz)
    t_common = data['t_common']
    N = len(t_common)

    with open(scaffold_json) as f:
        info = json.load(f)
    segments = [(s[0], s[1], s[2]) for s in info.get('segments', [])]

    state_path = None
    if os.path.exists(rslds_npz):
        rslds = np.load(rslds_npz)
        state_path = rslds['path']

    # Load raw EEG
    xdf_files = glob.glob(os.path.join(RAW_DIR, f'{session_name}*.xdf'))
    if not xdf_files:
        print(f"  No XDF for {session_name}")
        return None

    session_data = load_xdf_session(xdf_files[0])
    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions
                     if session_name.lower() in n.lower()]
    if not cache_matches:
        print(f"  No cache for {session_name}")
        return None

    cached = load_session_from_cache(cache_matches[0], config)

    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = (float(lsl_ts_p1[0])
                  - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0]))

    # Determine role for sign convention
    p1_role = session_data.get('p1_role', 'unknown')
    asym_sign = -1.0 if p1_role == 'patient' else 1.0

    # Compute directed coupling
    result = eeg_directed_burst_coupling(
        cached, t_common, lsl_offset,
        max_lag_samples=4, k_te=3, n_surrogates=200, seed=42)

    if result is None:
        print(f"  No EEG data for {session_name}")
        return None

    # Per-condition analysis (TE asymmetry + episode concentration)
    cond_results = {}
    for cond_name, t0, t1 in segments:
        mask = (t_common >= t0) & (t_common <= t1)
        n_mask = mask.sum()
        if n_mask < 10:
            continue
        cond_results[cond_name] = {}
        for band_name, bd in result.items():
            tc_eca = bd['eca_asym_tc']
            tc_te = bd['te_asym_tc']
            tc_te_z = bd.get('te_z_tc', np.zeros_like(tc_te))

            # Count directed episode samples within this condition
            t_eps = bd.get('therapist_leads_episodes', [])
            p_eps = bd.get('patient_leads_episodes', [])
            t_samp = sum(min(e['offset'], len(mask)) - e['onset']
                         for e in t_eps
                         if any(mask[max(0,e['onset']):min(e['offset'],len(mask))]))
            p_samp = sum(min(e['offset'], len(mask)) - e['onset']
                         for e in p_eps
                         if any(mask[max(0,e['onset']):min(e['offset'],len(mask))]))

            # More precise: count only samples within condition AND episode
            t_in_cond = 0
            for e in t_eps:
                ep_mask = np.zeros(len(mask), dtype=bool)
                ep_mask[e['onset']:e['offset']] = True
                t_in_cond += (ep_mask & mask).sum()
            p_in_cond = 0
            for e in p_eps:
                ep_mask = np.zeros(len(mask), dtype=bool)
                ep_mask[e['onset']:e['offset']] = True
                p_in_cond += (ep_mask & mask).sum()

            cond_results[cond_name][band_name] = {
                'eca_asym': float(tc_eca[mask].mean()) * asym_sign,
                'te_asym': float(tc_te[mask].mean()) * asym_sign,
                'te_z_mean': float(tc_te_z[mask].mean()) * asym_sign,
                'frac_therapist_leads': float(t_in_cond / n_mask),
                'frac_patient_leads': float(p_in_cond / n_mask),
            }

    # Per-state analysis
    state_results = {}
    if state_path is not None:
        for si, slabel in enumerate(state_labels):
            smask = state_path == si
            n_smask = smask.sum()
            if n_smask < 10:
                continue
            state_results[slabel] = {}
            for band_name, bd in result.items():
                tc_te_z = bd.get('te_z_tc', np.zeros_like(bd['te_asym_tc']))
                t_eps = bd.get('therapist_leads_episodes', [])
                p_eps = bd.get('patient_leads_episodes', [])

                t_in_state = 0
                for e in t_eps:
                    ep_mask = np.zeros(len(smask), dtype=bool)
                    ep_mask[e['onset']:e['offset']] = True
                    t_in_state += (ep_mask & smask).sum()
                p_in_state = 0
                for e in p_eps:
                    ep_mask = np.zeros(len(smask), dtype=bool)
                    ep_mask[e['onset']:e['offset']] = True
                    p_in_state += (ep_mask & smask).sum()

                state_results[slabel][band_name] = {
                    'eca_asym': float(bd['eca_asym_tc'][smask].mean()) * asym_sign,
                    'te_asym': float(bd['te_asym_tc'][smask].mean()) * asym_sign,
                    'te_z_mean': float(tc_te_z[smask].mean()) * asym_sign,
                    'frac_therapist_leads': float(t_in_state / n_smask),
                    'frac_patient_leads': float(p_in_state / n_smask),
                    'n_samples': int(n_smask),
                }

    # Session summary (role-corrected)
    band_summary = {}
    for band_name, bd in result.items():
        t_eps = bd.get('therapist_leads_episodes', [])
        p_eps = bd.get('patient_leads_episodes', [])
        band_summary[band_name] = {
            'eca_asymmetry': bd['eca_asymmetry'] * asym_sign,
            'eca_p1_leads': bd.get('eca_p1_leads', []),
            'eca_p2_leads': bd.get('eca_p2_leads', []),
            'eca_lags_s': bd.get('eca_lags_s', []),
            'te_p1_to_p2': bd['te_p1_to_p2'],
            'te_p2_to_p1': bd['te_p2_to_p1'],
            'te_asymmetry': bd['te_asymmetry'] * asym_sign,
            'te_z': bd['te_z'],
            'n_valid_channels': bd['n_valid_channels'],
            'n_p1_onsets': bd.get('n_p1_onsets', 0),
            'n_p2_onsets': bd.get('n_p2_onsets', 0),
            'n_therapist_episodes': len(t_eps),
            'n_patient_episodes': len(p_eps),
            'frac_therapist_leads': bd.get('frac_therapist_leads', 0),
            'frac_patient_leads': bd.get('frac_patient_leads', 0),
            'mean_episode_dur_s': float(np.mean(
                [e['duration_s'] for e in t_eps + p_eps])) if t_eps + p_eps else 0,
        }

    return {
        'session': session_name,
        'p1_role': p1_role,
        'asym_sign': asym_sign,
        'n_timepoints': N,
        'per_condition': cond_results,
        'per_state': state_results,
        'per_band_summary': band_summary,
    }


def main(session_name=None):
    print("=" * 80)
    print("  V11 Post-Hoc: Directed EEG Burst Coupling (ECA + TE)")
    print("  (positive = therapist leads, negative = patient leads)")
    print("=" * 80)

    t_wall = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    config = load_config()

    # Load state label permutation from the V11 hierarchical fit
    state_labels = _load_state_labels()
    print(f"  State labels (index order): {state_labels}")

    if session_name:
        session_ids = [session_name]
    else:
        npzs = sorted(glob.glob(f'{SCAFFOLD_DIR}/*/scaffold_v11_ztimecourses.npz'))
        session_ids = [os.path.basename(os.path.dirname(p)) for p in npzs
                       if os.path.basename(os.path.dirname(p)) != 'hierarchical']

    print(f"\n  {len(session_ids)} sessions to analyze")

    all_results = []
    for sid in session_ids:
        print(f"\n  {sid}...", end=' ', flush=True)
        t0 = time.time()
        try:
            result = analyze_session(sid, config, state_labels)
            if result is not None:
                all_results.append(result)
                bands = list(result['per_band_summary'].keys())
                te_z = [abs(result['per_band_summary'][b]['te_z']) for b in bands]
                n_eps = sum(result['per_band_summary'][b]['n_therapist_episodes']
                            + result['per_band_summary'][b]['n_patient_episodes']
                            for b in bands)
                print(f"{time.time()-t0:.1f}s  TE_z={max(te_z):.2f}  {n_eps} directed episodes")
            else:
                print("skipped")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    if not all_results:
        print("No results.")
        return

    # ── Per-condition summary ───────────────────────────────────────
    band_names = list(EEG_BANDS.keys())

    print(f"\n{'='*80}")
    print(f"  Per-Condition Directed ECA Asymmetry (+ = therapist leads)")
    print(f"{'='*80}")
    print(f"\n  {'Condition':>12s} |", end='')
    for bn in band_names:
        print(f" {bn:>8s}", end='')
    print(f" | n")
    print(f"  " + "-" * (15 + 9 * len(band_names) + 5))

    for cond in CONDS:
        vals = {bn: [] for bn in band_names}
        for r in all_results:
            if cond in r['per_condition']:
                for bn in band_names:
                    if bn in r['per_condition'][cond]:
                        vals[bn].append(r['per_condition'][cond][bn]['eca_asym'])
        n = max(len(v) for v in vals.values()) if any(vals.values()) else 0
        if n < 1:
            continue
        print(f"  {cond:>12s} |", end='')
        for bn in band_names:
            if vals[bn]:
                print(f" {np.mean(vals[bn]):>+8.4f}", end='')
            else:
                print(f" {'':>8s}", end='')
        print(f" | {n}")

    print(f"\n{'='*80}")
    print(f"  Per-Condition Directed TE Asymmetry (+ = therapist leads)")
    print(f"{'='*80}")
    print(f"\n  {'Condition':>12s} |", end='')
    for bn in band_names:
        print(f" {bn:>8s}", end='')
    print(f" | n")
    print(f"  " + "-" * (15 + 9 * len(band_names) + 5))

    for cond in CONDS:
        vals = {bn: [] for bn in band_names}
        for r in all_results:
            if cond in r['per_condition']:
                for bn in band_names:
                    if bn in r['per_condition'][cond]:
                        vals[bn].append(r['per_condition'][cond][bn]['te_asym'])
        n = max(len(v) for v in vals.values()) if any(vals.values()) else 0
        if n < 1:
            continue
        print(f"  {cond:>12s} |", end='')
        for bn in band_names:
            if vals[bn]:
                print(f" {np.mean(vals[bn]):>+8.5f}", end='')
            else:
                print(f" {'':>8s}", end='')
        print(f" | {n}")

    # ── Per-state summary ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Per-rSLDS-State ECA Asymmetry")
    print(f"{'='*80}")
    # Display in a fixed reading order regardless of the hierarchical fit's
    # index permutation, so output is comparable across refits.
    DISPLAY_ORDER = ['NULL', 'COUP', 'SHARED', 'OTHER']
    state_display = [s for s in DISPLAY_ORDER if s in state_labels]
    for slabel in state_display:
        vals = {bn: [] for bn in band_names}
        for r in all_results:
            if slabel in r.get('per_state', {}):
                for bn in band_names:
                    if bn in r['per_state'][slabel]:
                        vals[bn].append(r['per_state'][slabel][bn]['eca_asym'])
        n = max(len(v) for v in vals.values()) if any(vals.values()) else 0
        if n < 1:
            continue
        print(f"  {slabel:>8s} |", end='')
        for bn in band_names:
            if vals[bn]:
                print(f" {np.mean(vals[bn]):>+8.4f}", end='')
            else:
                print(f" {'':>8s}", end='')
        print(f" | {n}")

    # ── Directed episodes per condition ─────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Directed TE Episodes (% time in therapist-leads / patient-leads)")
    print(f"{'='*80}")
    print(f"\n  {'Condition':>12s} |", end='')
    for bn in band_names:
        print(f"  {bn:>5s}_T  {bn:>5s}_P", end='')
    print(f" | n")
    print(f"  " + "-" * (15 + 16 * len(band_names) + 5))

    for cond in CONDS:
        vals_t = {bn: [] for bn in band_names}
        vals_p = {bn: [] for bn in band_names}
        for r in all_results:
            if cond in r['per_condition']:
                for bn in band_names:
                    if bn in r['per_condition'][cond]:
                        vals_t[bn].append(r['per_condition'][cond][bn].get('frac_therapist_leads', 0))
                        vals_p[bn].append(r['per_condition'][cond][bn].get('frac_patient_leads', 0))
        n = max(len(v) for v in vals_t.values()) if any(vals_t.values()) else 0
        if n < 1:
            continue
        print(f"  {cond:>12s} |", end='')
        for bn in band_names:
            if vals_t[bn]:
                mt = np.mean(vals_t[bn]) * 100
                mp = np.mean(vals_p[bn]) * 100
                print(f"  {mt:>5.1f}%  {mp:>5.1f}%", end='')
            else:
                print(f"  {'':>5s}   {'':>5s} ", end='')
        print(f" | {n}")

    # ── Episode summary per band ─────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Directed Episode Summary (z > 2, >= 3s, 5s merge)")
    print(f"{'='*80}")
    for bn in band_names:
        n_t = [r['per_band_summary'][bn].get('n_therapist_episodes', 0) for r in all_results]
        n_p = [r['per_band_summary'][bn].get('n_patient_episodes', 0) for r in all_results]
        dur = [r['per_band_summary'][bn].get('mean_episode_dur_s', 0) for r in all_results]
        dur_valid = [d for d in dur if d > 0]
        print(f"  {bn:>6s}: therapist-leads={np.mean(n_t):.1f} eps/session, "
              f"patient-leads={np.mean(n_p):.1f} eps/session"
              f"  mean_dur={np.mean(dur_valid):.1f}s" if dur_valid else "")

    # ── Save ────────────────────────────────────────────────────────
    # Strip numpy arrays for JSON serialization
    def _clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    # Remove timecourse arrays from JSON (too large)
    save_results = []
    for r in all_results:
        sr = _clean(r)
        save_results.append(sr)

    output = {
        'version': 'v11_directed_burst_coupling',
        'n_sessions': len(all_results),
        'max_lag_samples': 4,
        'te_history_k': 3,
        'n_surrogates': 200,
        'sessions': save_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = os.path.join(OUT_DIR, 'directed_coupling_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    total = time.time() - t_wall
    print(f"\n  Total: {total:.0f}s")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V11 Directed Burst Coupling')
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        main()
    else:
        main(session_name=args.session or 'y_06')
