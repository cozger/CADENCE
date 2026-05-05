"""V11 Post-Hoc: EEG Burst Coincidence Analysis.

Temporally precise inter-brain coupling from native-rate EEG bursts.
Post-hoc layer on V10 scaffold — loads raw EEG from cache, computes
surrogate-calibrated burst coincidence per band, reports per-condition
and per-rSLDS-state coincidence rates.

Usage:
    python scripts/run_burst_coincidence_analysis.py                # y_06 only
    python scripts/run_burst_coincidence_analysis.py --session y_17 # specific
    python scripts/run_burst_coincidence_analysis.py --all          # all sessions
"""

import sys, os, time, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.significance.burst_coincidence import eeg_burst_coincidence, EEG_BANDS
from cadence.significance.coupling_bursts import event_coincidence_analysis
from scripts.run_session_v6 import load_xdf_session

RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
SCAFFOLD_DIR = 'results/v11'
OUT_DIR = 'results/v11/burst_coincidence'
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
    """Compute burst coincidence for one session.

    Args:
        state_labels: index-ordered labels from the hierarchical fit.
            state_labels[si] is the label of rSLDS state with index si in
            results/v11/{session}/v11_rslds_results.npz['path'].
    """

    # ── Load scaffold results ───────────────────────────────────────
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

    # rSLDS state path (optional — skip state analysis if missing)
    state_path = None
    if os.path.exists(rslds_npz):
        rslds = np.load(rslds_npz)
        state_path = rslds['path']

    # ── Load raw EEG from cache ─────────────────────────────────────
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

    # LSL offset (same formula as V10 scaffold)
    lsl_ts_p1 = session_data['landmarks']['P1'][0]
    lsl_offset = (float(lsl_ts_p1[0])
                  - float(cached.get('p1_blendshapes_ts', lsl_ts_p1)[0]))

    # ── Compute burst coincidence ───────────────────────────────────
    result = eeg_burst_coincidence(
        cached, t_common, lsl_offset,
        tau_samples=1, n_surrogates=200, seed=42)

    if result is None:
        print(f"  No EEG data for {session_name}")
        return None

    # ── Per-condition coincidence ────────────────────────────────────
    cond_results = {}
    for cond_name, t0, t1 in segments:
        mask = (t_common >= t0) & (t_common <= t1)
        if mask.sum() < 10:
            continue
        cond_results[cond_name] = {}
        for band_name, bd in result.items():
            cond_results[cond_name][band_name] = {
                'mean_z': float(bd['z'][mask].mean()),
                'mean_raw': float(bd['raw'][mask].mean()),
            }

    # ── Per-rSLDS-state coincidence ─────────────────────────────────
    state_results = {}
    if state_path is not None:
        for si, slabel in enumerate(state_labels):
            smask = state_path == si
            if smask.sum() < 10:
                continue
            state_results[slabel] = {}
            for band_name, bd in result.items():
                state_results[slabel][band_name] = {
                    'mean_z': float(bd['z'][smask].mean()),
                    'mean_raw': float(bd['raw'][smask].mean()),
                    'n_samples': int(smask.sum()),
                }

    # ── ECA: burst coincidence events vs coupling excess events ─────
    # Detect coincidence events (z > 2) and test against EEG Phase
    # coupling excess events for temporal co-occurrence
    eca_results = {}
    for band_name, bd in result.items():
        z = bd['z']
        # Detect coincidence events (sustained z > 2 for ≥ 2s)
        above = z > 2.0
        events = []
        in_event = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_event:
                start = i
                in_event = True
            elif not above[i] and in_event:
                if (i - start) / FS_OUT >= 2.0:
                    peak_idx = start + int(np.argmax(z[start:i]))
                    events.append({
                        'onset': int(start), 'offset': int(i),
                        'peak_idx': peak_idx, 'peak_z': float(z[peak_idx]),
                        'duration_s': float((i - start) / FS_OUT),
                        'direction': 'excitation',
                    })
                in_event = False
        if in_event and (len(above) - start) / FS_OUT >= 2.0:
            peak_idx = start + int(np.argmax(z[start:]))
            events.append({
                'onset': int(start), 'offset': int(len(above)),
                'peak_idx': peak_idx, 'peak_z': float(z[peak_idx]),
                'duration_s': float((len(above) - start) / FS_OUT),
                'direction': 'excitation',
            })

        dur_min = (t_common[-1] - t_common[0]) / 60.0
        eca_results[band_name] = {
            'n_events': len(events),
            'rate_per_min': len(events) / dur_min if dur_min > 0 else 0,
            'mean_z': float(z.mean()),
            'n_valid_channels': result[band_name]['n_valid_channels'],
        }

    # Compile
    return {
        'session': session_name,
        'n_timepoints': N,
        'per_condition': cond_results,
        'per_state': state_results,
        'per_band_summary': eca_results,
    }


def main(session_name=None):
    print("=" * 80)
    print("  V11 Post-Hoc: EEG Burst Coincidence Analysis")
    print("  (Surrogate-calibrated, tau=500ms, 200 surrogates)")
    print("=" * 80)

    t_wall = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    config = load_config()

    # Load state label permutation from the V11 hierarchical fit
    state_labels = _load_state_labels()
    print(f"  State labels (index order): {state_labels}")

    # Find sessions with V11 scaffold
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
                n_ev = sum(result['per_band_summary'][b]['n_events'] for b in bands)
                print(f"{time.time()-t0:.1f}s, {n_ev} coincidence events")
            else:
                print("skipped")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    if not all_results:
        print("No results.")
        return

    # ── Per-condition summary (cross-session) ───────────────────────
    print(f"\n{'='*80}")
    print(f"  Per-Condition Burst Coincidence (mean z across sessions)")
    print(f"{'='*80}")

    band_names = list(EEG_BANDS.keys())
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
                        vals[bn].append(r['per_condition'][cond][bn]['mean_z'])
        n = max(len(v) for v in vals.values()) if any(vals.values()) else 0
        if n < 1:
            continue
        print(f"  {cond:>12s} |", end='')
        for bn in band_names:
            if vals[bn]:
                print(f" {np.mean(vals[bn]):>+8.3f}", end='')
            else:
                print(f" {'':>8s}", end='')
        print(f" | {n}")

    # ── Per-state summary ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Per-rSLDS-State Burst Coincidence (mean z)")
    print(f"{'='*80}")

    print(f"\n  {'State':>8s} |", end='')
    for bn in band_names:
        print(f" {bn:>8s}", end='')
    print(f" | n")
    print(f"  " + "-" * (11 + 9 * len(band_names) + 5))

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
                        vals[bn].append(r['per_state'][slabel][bn]['mean_z'])
        n = max(len(v) for v in vals.values()) if any(vals.values()) else 0
        if n < 1:
            continue
        print(f"  {slabel:>8s} |", end='')
        for bn in band_names:
            if vals[bn]:
                print(f" {np.mean(vals[bn]):>+8.3f}", end='')
            else:
                print(f" {'':>8s}", end='')
        print(f" | {n}")

    # ── Per-band event rates ────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Burst Coincidence Event Rates (z > 2, >= 2s)")
    print(f"{'='*80}")

    for bn in band_names:
        rates = [r['per_band_summary'][bn]['rate_per_min']
                 for r in all_results if bn in r['per_band_summary']]
        mean_z = [r['per_band_summary'][bn]['mean_z']
                  for r in all_results if bn in r['per_band_summary']]
        n_ch = [r['per_band_summary'][bn]['n_valid_channels']
                for r in all_results if bn in r['per_band_summary']]
        if rates:
            print(f"  {bn:>6s}: {np.mean(rates):.2f}+/-{np.std(rates)/np.sqrt(len(rates)):.2f} events/min"
                  f"  mean_z={np.mean(mean_z):+.3f}"
                  f"  valid_ch={np.mean(n_ch):.0f}")

    # ── Save ────────────────────────────────────────────────────────
    output = {
        'version': 'v11_burst_coincidence',
        'n_sessions': len(all_results),
        'tau_samples': 1,
        'n_surrogates': 200,
        'sessions': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = os.path.join(OUT_DIR, 'burst_coincidence_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    total = time.time() - t_wall
    print(f"\n  Total: {total:.0f}s")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V11 Burst Coincidence Analysis')
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        main()
    else:
        main(session_name=args.session or 'y_06')
