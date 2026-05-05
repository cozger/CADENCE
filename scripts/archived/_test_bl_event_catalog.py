"""Event catalog: precise timestamps for video verification.

For each significant session, outputs the exact times of detected mimicry
events relative to the XDF recording start, so you can check them in the video.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyxdf
from cadence.significance.bl_coupling import bl_two_stage_coupling
from cadence.ingest.roles import roles_dict_from_session as _detect_roles

FS = 30.0
RAW_DIR = r'C:\Users\optilab\Desktop\CADENCE\raw sessions'
HETERO_AUS = [0, 1, 2, 3, 5, 6, 7, 18, 19, 20, 27, 28, 29, 30, 43, 44, 49, 50]

# Sessions with strongest signals
SESSIONS = [
    ('y05_02192026.xdf', 'conv_2'),  # T→P smile p=0.01
    ('y_32_03132026.xdf', 'conv_2'), # T→P smile p=0.03, P→T brow p=0.04
    ('y_06.xdf', 'conv_1'),          # T→P trending
    ('y_06.xdf', 'conv_2'),          # for comparison
    ('y01_021726.xdf', 'conv_2'),    # T→P brow p=0.04
]


def load_segment(xdf_path, conv_label):
    data, _ = pyxdf.load_xdf(xdf_path)
    roles = _detect_roles(data)
    p1_role = roles['p1_role']
    p2_role = roles['p2_role']

    # Get absolute marker times (for video timestamp reference)
    markers = {}
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                markers[v[0]] = t

    t_start_abs = markers.get(f'{conv_label}_start')
    t_end_abs = markers.get(f'{conv_label}_stop')
    if t_start_abs is None:
        return None

    # Also get recording start for absolute timestamps
    rec_start = min(s['time_stamps'][0] for s in data if len(s['time_stamps']) > 0)

    landmarks = {}
    for stream in data:
        name = stream['info']['name'][0]
        if 'landmarks' in name.lower():
            person = 'P1' if 'P1' in name else 'P2'
            n_ch = int(stream['info']['channel_count'][0])
            if n_ch >= 52 and person not in landmarks:
                landmarks[person] = (np.array(stream['time_stamps']),
                                     np.array(stream['time_series'], dtype=np.float32))

    if 'P1' not in landmarks or 'P2' not in landmarks:
        return None

    ts_p1, d_p1 = landmarks['P1']
    ts_p2, d_p2 = landmarks['P2']
    m1 = (ts_p1 >= t_start_abs) & (ts_p1 <= t_end_abs)
    m2 = (ts_p2 >= t_start_abs) & (ts_p2 <= t_end_abs)

    dur = min(ts_p1[m1][-1], ts_p2[m2][-1]) - t_start_abs
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)

    sigs = {}
    for label, ts, d, m in [('P1', ts_p1, d_p1, m1), ('P2', ts_p2, d_p2, m2)]:
        ts_l = ts[m] - t_start_abs
        sig = np.stack([np.interp(t_grid, ts_l, d[m, c]) for c in range(52)], 1)
        # Keep raw — bl_two_stage_coupling z-scores internally for Stage 1
        sigs[label] = sig

    return {
        'p1': sigs['P1'], 'p2': sigs['P2'], 'T': T, 'dur': dur,
        'p1_role': p1_role, 'p2_role': p2_role,
        'conv_start_abs': t_start_abs,
        'rec_start': rec_start,
        'offset_from_rec_start': t_start_abs - rec_start,
    }


def fmt_time(seconds):
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


if __name__ == '__main__':
    for xdf_name, conv in SESSIONS:
        xdf_path = os.path.join(RAW_DIR, xdf_name)
        session_name = os.path.splitext(xdf_name)[0]

        print(f"\n{'='*80}")
        print(f"SESSION: {session_name} / {conv}")
        print(f"{'='*80}")

        info = load_segment(xdf_path, conv)
        if info is None:
            print("  Could not load segment")
            continue

        offset = info['offset_from_rec_start']
        print(f"  P1={info['p1_role']}, P2={info['p2_role']}")
        print(f"  Conv starts at {fmt_time(offset)} from recording start")
        print(f"  Duration: {info['dur']:.0f}s")

        # Run both directions
        for src_label, tgt_label, src, tgt in [('P1', 'P2', info['p1'], info['p2']),
                                                 ('P2', 'P1', info['p2'], info['p1'])]:
            src_role = info['p1_role'] if src_label == 'P1' else info['p2_role']
            tgt_role = info['p1_role'] if tgt_label == 'P1' else info['p2_role']

            res = bl_two_stage_coupling(
                src, tgt, FS,
                composites=None,
                xcorr_channels=HETERO_AUS,
                max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
                n_surrogates=100, target_fa=0.05,
                event_prominence=4.0, min_event_iei_s=3.0,
                response_threshold=2.0,
                n_surrogates_event=500,
                population_prior=(2.5, 1.0), seed=42)

            # Only show composites with p < 0.3
            interesting = {k: v for k, v in res.composites.items()
                          if v.session_p_value < 0.3 and v.n_matched > 0}
            if not interesting:
                continue

            print(f"\n  --- {src_role} -> {tgt_role} ---")
            print(f"  Lag window: [{res.lag_window_s[0]:.1f}, {res.lag_window_s[1]:.1f}]s "
                  f"(estimated lag: {res.estimated_lag_s:.2f}s)")

            for comp_name, cr in sorted(interesting.items(),
                                         key=lambda x: x[1].session_p_value):
                sig = '**' if cr.session_p_value < 0.01 else (
                      '*' if cr.session_p_value < 0.05 else '')
                print(f"\n  Composite: {comp_name} — "
                      f"{cr.n_matched}/{cr.n_events} matched "
                      f"({cr.match_rate:.0%} vs null {cr.null_match_rate:.0%}) "
                      f"p={cr.session_p_value:.3f}{sig}")
                if cr.mean_lag is not None:
                    print(f"  Mean response lag: {cr.mean_lag:.2f}s"
                          + (f" (std={cr.lag_std:.2f}s)" if cr.lag_std else ""))

                # Event table with LSL timestamps
                conv_start_lsl = info['conv_start_abs']
                print(f"\n  {'#':>3} {'Src(conv)':>10} {'Src(LSL)':>16} "
                      f"{'Amp':>5} {'Match':>6} {'Tgt(LSL)':>16} "
                      f"{'Lag':>5}")

                for i, e in enumerate(cr.events):
                    src_lsl = e.source_time + conv_start_lsl
                    amp = e.source_amplitude

                    if e.matched:
                        tgt_lsl = e.target_time + conv_start_lsl
                        lag = e.lag
                        mark = '<-- MIMICRY'
                        print(f"  {i+1:>3} {fmt_time(e.source_time):>10} "
                              f"{src_lsl:>16.3f} "
                              f"{amp:>5.1f} {'YES':>6} "
                              f"{tgt_lsl:>16.3f} "
                              f"{lag:>5.2f}  {mark}")
                    else:
                        print(f"  {i+1:>3} {fmt_time(e.source_time):>10} "
                              f"{src_lsl:>16.3f} "
                              f"{amp:>5.1f} {'':>6}")

    print("\nDone.")
