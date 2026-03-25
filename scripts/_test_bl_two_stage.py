"""Two-stage BL coupling pipeline — real data validation (y_06).

Tests the complete pipeline on real conversation data:
Stage 1: Cross-product → coupling windows + lag estimation
Stage 2: Per-event mimicry with tight lag window → per-event p-values
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyxdf
from cadence.significance.bl_coupling import bl_two_stage_coupling

FS = 30.0


def load_conversation_bl(xdf_path, conv_label):
    """Load P1 and P2 blendshapes for a conversation section."""
    data, _ = pyxdf.load_xdf(xdf_path)

    # Get conversation boundaries
    t_start = t_end = None
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                if f'{conv_label}_start' in v[0]:
                    t_start = t
                if f'{conv_label}_stop' in v[0]:
                    t_end = t

    if t_start is None or t_end is None:
        raise ValueError(f"Could not find markers for {conv_label}")

    # Load landmarks
    signals = {}
    for stream in data:
        name = stream['info']['name'][0]
        if 'landmarks' in name.lower():
            person = 'P1' if 'P1' in name else 'P2'
            ts = np.array(stream['time_stamps'])
            d = np.array(stream['time_series'], dtype=np.float32)
            mask = (ts >= t_start) & (ts <= t_end)
            signals[person] = (ts[mask] - t_start, d[mask, :52])

    # Common 30 Hz grid
    dur = min(signals['P1'][0][-1], signals['P2'][0][-1])
    T = int(dur * FS)
    t_grid = np.linspace(0, dur, T)

    result = {}
    for person in ['P1', 'P2']:
        ts, d = signals[person]
        sig = np.stack([np.interp(t_grid, ts, d[:, c]) for c in range(52)], axis=1)
        # Z-score per channel
        for c in range(52):
            mu, sd = sig[:, c].mean(), max(sig[:, c].std(), 1e-8)
            sig[:, c] = (sig[:, c] - mu) / sd
        result[person] = sig

    return result['P1'], result['P2'], T, dur


def print_result(res, direction):
    """Print two-stage coupling result."""
    sig = '**' if res.session_p_value < 0.01 else ('*' if res.session_p_value < 0.05 else '')

    print(f"\n  {direction}:")
    print(f"    Stage 1 (continuous): coupling_frac={res.mask_continuous.mean():.1%} "
          f"z_mean={res.z_continuous.mean():.2f} z_max={res.z_continuous.max():.2f}")
    print(f"    Lag estimate: {res.estimated_lag_s:.2f}s "
          f"(confidence={res.lag_confidence:.2f}, "
          f"window=[{res.lag_window_s[0]:.1f}, {res.lag_window_s[1]:.1f}]s)")
    print(f"    Stage 2 (per-event): {res.n_events} events, "
          f"{res.n_matched} matched ({res.match_rate:.0%}) "
          f"vs null {res.null_match_rate:.0%} "
          f"→ p={res.session_p_value:.4f}{sig}")
    if res.mean_lag is not None and res.lag_std is not None:
        print(f"    Response lags: mean={res.mean_lag:.2f}s "
              f"median={res.median_lag:.2f}s std={res.lag_std:.2f}s")
    elif res.mean_lag is not None:
        print(f"    Response lag: {res.mean_lag:.2f}s (single match)")
    print(f"    Per-event null p: {res.diagnostics['p_null_single_event']:.3f} "
          f"(P2 rate={res.diagnostics['p2_event_rate']:.3f}/s, "
          f"window={res.diagnostics['window_width_s']:.1f}s)")

    # Print individual events
    print(f"\n    {'P1_time':>8} {'P1_amp':>7} {'Matched':>8} {'P2_time':>8} "
          f"{'Lag':>5} {'p_event':>8}")
    for e in res.events:
        m_str = 'YES' if e.matched else ''
        t2_str = f"{e.target_time:.1f}s" if e.target_time else ''
        lag_str = f"{e.lag:.2f}" if e.lag else ''
        p_str = f"{e.p_value:.3f}" if e.matched else ''
        print(f"    {e.source_time:>7.1f}s {e.source_amplitude:>7.2f} "
              f"{m_str:>8} {t2_str:>8} {lag_str:>5} {p_str:>8}")


if __name__ == '__main__':
    xdf_path = r'C:\Users\optilab\Desktop\CADENCE\raw sessions\y_06.xdf'

    for conv in ['conv_1', 'conv_2']:
        print(f"\n{'='*70}")
        print(f"Session y_06 — {conv}")
        print(f"{'='*70}")

        p1, p2, T, dur = load_conversation_bl(xdf_path, conv)
        print(f"  Duration: {dur:.0f}s, T={T}, P1: {p1.shape}, P2: {p2.shape}")

        # HETERO_AUS for Stage 1 cross-product
        hetero_aus = [0, 1, 2, 3, 5, 6, 7, 18, 19, 20, 27, 28, 29, 30, 43, 44, 49, 50]

        # Run both directions
        for src_name, tgt_name, src, tgt in [('P1', 'P2', p1, p2),
                                               ('P2', 'P1', p2, p1)]:
            t0 = time.perf_counter()
            res = bl_two_stage_coupling(
                src, tgt, FS,
                composite_aus=(43, 44, 17),
                xcorr_channels=hetero_aus,
                max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
                n_surrogates=100, target_fa=0.05,
                event_prominence=4.0, min_event_iei_s=3.0,
                response_threshold=2.0,
                n_surrogates_event=500, seed=42)
            elapsed = time.perf_counter() - t0

            print_result(res, f"{src_name}→{tgt_name} ({elapsed:.1f}s)")

    print("\nDone.")
