import pyxdf, glob
data, _ = pyxdf.load_xdf(glob.glob('C:/Users/optilab/Desktop/CADENCE/raw sessions/y_06.xdf')[0], dejitter_timestamps=True)
mt = {v[0]: t for s in data if s['info']['type'][0] == 'Markers' for t, v in zip(s['time_stamps'], s['time_series'])}
t0 = mt['conv_1_start']
print(f"conv_1_start LSL = {t0:.3f}")
for i, (t, z) in enumerate([(384.1, 5.30), (30.1, 5.08), (393.9, 4.03), (17.0, 2.77), (111.8, 2.34)]):
    print(f"  Peak {i+1}: segment_t={t:6.1f}s  z={z:.2f}  LSL={t0+t:.3f}")
