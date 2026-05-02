"""Check if the corrected Stage 2 catches the smile at conv_2 t=279s."""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')
import numpy as np, pyxdf, glob
from cadence.significance.bl_coupling import bl_two_stage_coupling

data, _ = pyxdf.load_xdf(glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0], dejitter_timestamps=True)
mt = {v[0]: t for s in data if s['info']['type'][0] == 'Markers' for t, v in zip(s['time_stamps'], s['time_series'])}
landmarks = {}
for stream in data:
    name = stream['info']['name'][0]
    if 'landmarks' in name.lower():
        person = 'P1' if 'P1' in name else 'P2'
        n_ch = int(stream['info']['channel_count'][0])
        if n_ch >= 52 and person not in landmarks:
            landmarks[person] = (np.array(stream['time_stamps']), np.array(stream['time_series'], dtype=np.float32))

t_start = mt['conv_2_start']
t_end = mt['conv_2_stop']
dur = t_end - t_start
T = int(dur * 30)
t_grid = np.linspace(0, dur, T)
sigs = {}
for p in ['P1', 'P2']:
    ts, d = landmarks[p]
    m = (ts >= t_start) & (ts <= t_end)
    sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c]) for c in range(52)], axis=1)

result = bl_two_stage_coupling(sigs['P1'], sigs['P2'], 30.0, seed=42)

cat = result.catalogs['smile']
print(f"Smile composite: (44, 45) = mouthSmileLeft + mouthSmileRight")
print(f"P1 events: {cat.n_events_a}, P2 events: {cat.n_events_b}")
print(f"Co-occurrences: {cat.n_cooccurrences} (p={cat.session_p_value:.3f})")
print(f"Lag window: {result.lag_window_s}")
print(f"Estimated lag: {result.estimated_lag_s:.2f}s")

print(f"\nP1 smile events:")
for ev in cat.events_a:
    lsl = t_start + ev.time
    marker = " <<<" if abs(ev.time - 279) < 5 else ""
    print(f"  t={ev.time:6.1f}s  amp={ev.amplitude:.3f}  LSL={lsl:.1f}{marker}")

print(f"\nP2 smile events:")
for ev in cat.events_b:
    lsl = t_start + ev.time
    marker = " <<<" if abs(ev.time - 280) < 5 else ""
    print(f"  t={ev.time:6.1f}s  amp={ev.amplitude:.3f}  LSL={lsl:.1f}{marker}")

print(f"\nCo-occurrences:")
for co in cat.cooccurrences:
    lsl_a = t_start + co.person_a_time
    lsl_b = t_start + co.person_b_time
    marker = " <<<" if abs(co.person_a_time - 279) < 5 else ""
    print(f"  A={co.person_a_time:.1f}s B={co.person_b_time:.1f}s "
          f"lag={co.lag:.1f}s leader={co.leader} "
          f"attr={co.attribution} conf={co.causal_confidence:.2f}{marker}")
