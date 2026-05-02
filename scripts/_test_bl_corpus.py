"""Two-stage BL coupling across all sessions — role-aware (therapist/patient).

Extracts conv_1 and conv_2 from every XDF file, detects therapist/patient
roles from stream metadata (Sway = therapist), runs the two-stage pipeline,
and aggregates by role direction (Therapist→Patient vs Patient→Therapist).
"""
import sys, os, time, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyxdf
from joblib import Parallel, delayed
from cadence.significance.bl_coupling import bl_two_stage_coupling
from cadence.ingest.roles import roles_dict_from_session as _detect_roles

FS = 30.0
RAW_DIR = r'C:\Users\optilab\Desktop\CADENCE\raw sessions'
HETERO_AUS = [0, 1, 2, 3, 5, 6, 7, 18, 19, 20, 27, 28, 29, 30, 43, 44, 49, 50]


def extract_conversations(xdf_path):
    """Extract conv segments with role detection from XDF."""
    session_name = os.path.splitext(os.path.basename(xdf_path))[0]
    try:
        data, _ = pyxdf.load_xdf(xdf_path)
    except Exception as e:
        print(f"  ERROR loading {session_name}: {e}")
        return []

    # Detect roles
    roles = _detect_roles(data)
    p1_role = roles.get('p1_role')
    p2_role = roles.get('p2_role')
    p1_name = roles.get('p1_name', 'unknown')

    # Skip if roles are undetected (both "unknown")
    if p1_name == 'unknown':
        print(f"  {session_name}: roles undetected, skipping")
        return []

    print(f"  {session_name}: P1={p1_role}, P2={p2_role}")

    # Find markers
    marker_times = {}
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                marker_times[v[0]] = t

    # Find landmark streams
    landmarks = {}
    for stream in data:
        name = stream['info']['name'][0]
        if 'landmarks' in name.lower():
            person = 'P1' if 'P1' in name else 'P2'
            n_ch = int(stream['info']['channel_count'][0])
            if n_ch >= 52 and person not in landmarks:
                landmarks[person] = (
                    np.array(stream['time_stamps']),
                    np.array(stream['time_series'], dtype=np.float32))

    if 'P1' not in landmarks or 'P2' not in landmarks:
        print(f"    missing landmarks, skipping")
        return []

    results = []
    for conv in ['conv_1', 'conv_2']:
        t_start = marker_times.get(f'{conv}_start')
        t_end = marker_times.get(f'{conv}_stop')
        if t_start is None or t_end is None:
            continue
        if t_end - t_start < 60:
            continue

        try:
            ts_p1, d_p1 = landmarks['P1']
            ts_p2, d_p2 = landmarks['P2']
            m1 = (ts_p1 >= t_start) & (ts_p1 <= t_end)
            m2 = (ts_p2 >= t_start) & (ts_p2 <= t_end)
            if m1.sum() < 100 or m2.sum() < 100:
                continue

            ts1_l = ts_p1[m1] - t_start
            ts2_l = ts_p2[m2] - t_start
            d1_l = d_p1[m1, :52]
            d2_l = d_p2[m2, :52]

            dur = min(ts1_l[-1], ts2_l[-1])
            T = int(dur * FS)
            t_grid = np.linspace(0, dur, T)

            sigs = {}
            for label, ts_l, d_l in [('P1', ts1_l, d1_l), ('P2', ts2_l, d2_l)]:
                sig = np.stack([np.interp(t_grid, ts_l, d_l[:, c])
                                for c in range(52)], axis=1)
                # Keep raw — bl_two_stage_coupling z-scores internally for Stage 1
                sigs[label] = sig

            results.append((session_name, conv, sigs['P1'], sigs['P2'],
                            T, dur, p1_role, p2_role))
        except Exception as e:
            print(f"    {conv}: ERROR {e}")

    return results


def run_one_segment(session_name, conv, p1, p2, T, dur, p1_role, p2_role,
                    lag_prior_s=None, population_prior=(2.5, 1.0)):
    """Run two-stage pipeline, return results keyed by role direction."""
    outputs = {}
    for src_label, tgt_label, src, tgt in [('P1', 'P2', p1, p2),
                                             ('P2', 'P1', p2, p1)]:
        src_role = p1_role if src_label == 'P1' else p2_role
        tgt_role = p1_role if tgt_label == 'P1' else p2_role
        role_dir = f"{src_role}->{tgt_role}"

        try:
            res = bl_two_stage_coupling(
                src, tgt, FS,
                composites=None,  # use all defaults (general + smile + brow + frown + speech)
                xcorr_channels=HETERO_AUS,
                max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
                n_surrogates=100, target_fa=0.05,
                event_prominence=4.0, min_event_iei_s=3.0,
                response_threshold=2.0,
                n_surrogates_event=500,
                lag_prior_s=lag_prior_s,
                population_prior=population_prior, seed=42)
            outputs[role_dir] = res
        except Exception as e:
            print(f"  {session_name}/{conv} {role_dir}: ERROR {e}")
            outputs[role_dir] = None

    return session_name, conv, dur, outputs


if __name__ == '__main__':
    xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
    print(f"Found {len(xdf_files)} XDF files\n")

    # Extract all conversation segments with roles
    print("Extracting conversation segments...")
    all_segments = []
    for xdf_path in xdf_files:
        segs = extract_conversations(xdf_path)
        all_segments.extend(segs)

    print(f"\nTotal segments with roles: {len(all_segments)}")
    for seg in all_segments:
        print(f"  {seg[0]:>20s} {seg[1]:>6s}  dur={seg[5]:.0f}s  "
              f"P1={seg[6]} P2={seg[7]}")

    # ── Sweep approaches ──
    configs = [
        ('data-driven (no prior)', None, None),
        ('shrinkage (2.5, 1.0)', None, (2.5, 1.0)),
        ('shrinkage (2.5, 0.75)', None, (2.5, 0.75)),
        ('shrinkage (2.5, 1.5)', None, (2.5, 1.5)),
        ('hard [1.5, 3.5]', (1.5, 3.5), None),
    ]

    from scipy.stats import chi2 as chi2_dist

    for config_name, hard_prior, pop_prior in configs:
        print(f"\n{'='*105}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*105}")

        t0 = time.perf_counter()
        results = Parallel(n_jobs=-1)(
            delayed(run_one_segment)(
                *seg, lag_prior_s=hard_prior,
                population_prior=pop_prior)
            for seg in all_segments)
        elapsed = time.perf_counter() - t0

        # Collect by role × composite
        role_comp_data = {}  # (role_dir, comp_name) → list
        for session_name, conv, dur, outputs in results:
            for role_dir, res in outputs.items():
                if res is None:
                    continue
                # Print best composite for this segment
                sig = '**' if res.session_p_value < 0.01 else (
                      '*' if res.session_p_value < 0.05 else '')
                win = f"[{res.lag_window_s[0]:.1f},{res.lag_window_s[1]:.1f}]"
                # Show all composites inline
                comp_strs = []
                for cn, cr in res.composites.items():
                    cs = '**' if cr.session_p_value < 0.01 else (
                         '*' if cr.session_p_value < 0.05 else '')
                    comp_strs.append(f"{cn}={cr.n_matched}/{cr.n_events}"
                                     f"({cr.match_rate:.0%}|{cr.null_match_rate:.0%})"
                                     f"p={cr.session_p_value:.2f}{cs}")

                print(f"  {session_name:>12} {conv:>6} {role_dir:>15} "
                      f"best={res.best_composite} "
                      f"p={res.session_p_value:.3f}{sig:>2} "
                      f"win={win}")
                for cs in comp_strs:
                    print(f"    {cs}")

                # Collect per-composite
                for cn, cr in res.composites.items():
                    key = (role_dir, cn)
                    if key not in role_comp_data:
                        role_comp_data[key] = []
                    role_comp_data[key].append({
                        'session': session_name, 'conv': conv,
                        'match_rate': cr.match_rate,
                        'null_rate': cr.null_match_rate,
                        'p_value': cr.session_p_value,
                        'lag': cr.mean_lag,
                        'n_events': cr.n_events,
                        'n_matched': cr.n_matched,
                    })

        # Aggregation by role × composite
        print(f"\n  --- Aggregation ({elapsed:.0f}s) ---")
        for role_dir in sorted(set(k[0] for k in role_comp_data)):
            print(f"\n  {role_dir}:")
            for comp_name in ['general', 'smile', 'brow', 'frown', 'speech']:
                key = (role_dir, comp_name)
                if key not in role_comp_data:
                    continue
                entries = role_comp_data[key]
                n = len(entries)
                mr = [e['match_rate'] for e in entries]
                nr = [e['null_rate'] for e in entries]
                pv = [e['p_value'] for e in entries]
                lags = [e['lag'] for e in entries if e['lag'] is not None]
                n_sig = sum(1 for p in pv if p < 0.05)
                tot_ev = sum(e['n_events'] for e in entries)
                tot_m = sum(e['n_matched'] for e in entries)

                chi2_val = -2 * np.sum(np.log(np.maximum(pv, 1e-10)))
                fisher_p = 1 - chi2_dist.cdf(chi2_val, 2 * n)
                fsig = '**' if fisher_p < 0.01 else ('*' if fisher_p < 0.05 else '')

                lag_str = f"lag={np.mean(lags):.1f}s" if lags else ""
                print(f"    {comp_name:>10}: rate={np.mean(mr):.0%} "
                      f"null={np.mean(nr):.0%} "
                      f"ex={np.mean(mr)-np.mean(nr):>+.0%} "
                      f"{tot_m}/{tot_ev} "
                      f"sig={n_sig}/{n} F={fisher_p:.4f}{fsig} {lag_str}")

    print("\nDone.")
