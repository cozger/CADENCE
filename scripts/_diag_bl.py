"""Diagnostic: trace BL coupling signal through the pipeline stages."""
import numpy as np
import sys, yaml
sys.path.insert(0, '.')

from cadence.synthetic import build_synthetic_session_v2
from cadence.constants import SYNTH_MODALITY_CONFIG_V2

# Build session with BL coupling at kappa=0.30
kd = {
    'eeg_wavelet': 0.0,
    'ecg_features_v2': 0.0,
    'blendshapes_v2': 0.30,
    'pose_features': 0.0,
}
session = build_synthetic_session_v2(1800, kd, seed=1)

p1_bl = session['p1_blendshapes_v2']
p2_bl = session['p2_blendshapes_v2']
rate = SYNTH_MODALITY_CONFIG_V2['blendshapes_v2']['hz']

print(f"BL shape: {p1_bl.shape}, rate: {rate} Hz")
print(f"Duration: {p1_bl.shape[0]/rate:.0f}s")

# 1. Raw coupling signal check
print("\n=== 1. Raw cross-correlation (lagged) ===")
lag_s = SYNTH_MODALITY_CONFIG_V2['blendshapes_v2']['lag_s']
lag_samples = int(round(lag_s * rate))
n_coupled = SYNTH_MODALITY_CONFIG_V2['blendshapes_v2']['n_coupled']
print(f"Expected lag: {lag_s}s = {lag_samples} samples, n_coupled={n_coupled}")
for ch in [0, 3, 5, 9, 10, 14, 20, 25, 30]:
    if ch < p1_bl.shape[1]:
        cc = np.corrcoef(p1_bl[:-lag_samples, ch], p2_bl[lag_samples:, ch])[0, 1]
        coupled = "COUPLED" if ch < n_coupled else "uncoupled"
        print(f"  ch{ch:2d} ({coupled:9s}): r={cc:.4f}")

# 2. Coupling gates
gates = session.get('coupling_gates', {})
print(f"\n=== 2. Coupling gates ===")
bl_gate = gates.get('blendshapes_v2', [])
if isinstance(bl_gate, np.ndarray):
    # It's a timeseries envelope
    on_mask = bl_gate > 0.5
    total_on = on_mask.sum() / rate
    duty = total_on / (p1_bl.shape[0] / rate)
    print(f"  blendshapes_v2: envelope, duty={duty:.2%}, total_on={total_on:.0f}s")
    # Find episodes
    diff = np.diff(on_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if on_mask[0]:
        starts = np.concatenate([[0], starts])
    if on_mask[-1]:
        ends = np.concatenate([ends, [len(on_mask)]])
    print(f"  Episodes: {len(starts)}")
    for i in range(min(5, len(starts))):
        print(f"    [{starts[i]/rate:.1f}s - {ends[i]/rate:.1f}s] ({(ends[i]-starts[i])/rate:.1f}s)")
else:
    print(f"  Type: {type(bl_gate)}, len={len(bl_gate)}")

# 3. Pre-grouping effect
print("\n=== 3. Pre-grouping (triplet PCA) ===")
from cadence.coupling.estimator import CouplingEstimator
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
est = CouplingEstimator(config)
grouped, cluster_map = est._pregroup_features(p1_bl, 'blendshapes_v2')
print(f"  Raw: {p1_bl.shape[1]} ch -> Grouped: {grouped.shape[1]} groups")
for gid, channels in sorted(cluster_map.items()):
    coupled_in_group = [c for c in channels if c < n_coupled]
    label = f"coupled:{len(coupled_in_group)}/{len(channels)}" if coupled_in_group else "uncoupled"
    print(f"  Group {gid:2d}: channels {channels} ({label})")

# 4. Cross-correlation after grouping (source groups -> target channels)
print("\n=== 4. Grouped source -> raw target correlation ===")
for gid in range(min(grouped.shape[1], 6)):
    channels = cluster_map[gid]
    coupled_in = [c for c in channels if c < n_coupled]
    for tch in [0, 5, 10]:
        if tch < p2_bl.shape[1]:
            cc = np.corrcoef(grouped[:-lag_samples, gid], p2_bl[lag_samples:, tch])[0, 1]
            if abs(cc) > 0.01:
                print(f"  Group{gid}(coupled:{len(coupled_in)}) -> P2_ch{tch}: r={cc:.4f}")

# 5. Target PCA effect
print("\n=== 5. Target PCA ===")
ds_cfg = config.get('doubly_sparse', {})
max_tgt_dim = ds_cfg.get('max_target_dim', 10)
tgt = p2_bl.copy()
tgt_centered = tgt - tgt.mean(axis=0)
U, S, Vh = np.linalg.svd(tgt_centered, full_matrices=False)
print(f"  Singular values (top 10): {S[:10].round(1)}")
var_explained = S**2 / (S**2).sum() * 100
print(f"  Variance explained (top 10): {var_explained[:10].round(1)}%")
print(f"  Cumulative var explained (top 10): {np.cumsum(var_explained[:10]).round(1)}%")

tgt_pca = (tgt_centered @ Vh[:max_tgt_dim].T).astype(np.float32)
print(f"  Target PCA shape: {tgt_pca.shape}")

# Check which PCs capture coupling signal
print("\n  Coupling signal in PCA space:")
for pc in range(min(10, max_tgt_dim)):
    best_cc = 0
    best_grp = -1
    for gid in range(min(4, grouped.shape[1])):
        cc = np.corrcoef(grouped[:-lag_samples, gid], tgt_pca[lag_samples:, pc])[0, 1]
        if abs(cc) > abs(best_cc):
            best_cc = cc
            best_grp = gid
    if abs(best_cc) > 0.01:
        print(f"  PC{pc}: best r={best_cc:.4f} (Group{best_grp})")

# 6. Correlation during coupling ON vs OFF
print("\n=== 6. ON vs OFF episodes ===")
if isinstance(bl_gate, np.ndarray):
    on_mask = bl_gate > 0.5
    off_mask = ~on_mask
    for gid in range(min(4, grouped.shape[1])):
        channels = cluster_map[gid]
        coupled_in = [c for c in channels if c < n_coupled]
        for tch in [0, 5]:
            combined_on = on_mask[:-lag_samples] & on_mask[lag_samples:]
            combined_off = off_mask[:-lag_samples] & off_mask[lag_samples:]
            if combined_on.sum() > 100 and combined_off.sum() > 100:
                cc_on = np.corrcoef(grouped[:-lag_samples, gid][combined_on],
                                    p2_bl[lag_samples:, tch][combined_on])[0, 1]
                cc_off = np.corrcoef(grouped[:-lag_samples, gid][combined_off],
                                     p2_bl[lag_samples:, tch][combined_off])[0, 1]
                if abs(cc_on) > 0.02 or abs(cc_off) > 0.02:
                    print(f"  Group{gid}(coupled:{len(coupled_in)}) -> ch{tch}: "
                          f"ON r={cc_on:.4f}, OFF r={cc_off:.4f}")

# 7. What the group lasso actually sees: design matrix shape
print("\n=== 7. Design matrix dimensions ===")
from cadence.basis.raised_cosine import RaisedCosineBasis
basis_cfg = config.get('basis', {}).get('layer1', {})
n_basis = basis_cfg.get('n_basis', 8)
print(f"  Source groups: {grouped.shape[1]}")
print(f"  Basis functions: {n_basis}")
print(f"  Design matrix cols (source): {grouped.shape[1] * n_basis}")
print(f"  AR order: {config.get('autoregressive', {}).get('order', 3)}")
ar_order = config.get('autoregressive', {}).get('order', 3)
# For multi-target, AR adds C_tgt * ar_order columns
print(f"  Target PCA dims: {max_tgt_dim}")
print(f"  AR cols: {max_tgt_dim * ar_order}")
total_cols = grouped.shape[1] * n_basis + max_tgt_dim * ar_order
print(f"  Total design matrix cols: {total_cols}")

block_dur = config.get('doubly_sparse', {}).get('block_duration', 120)
eval_rate = config.get('eval_rate', 5)
block_samples = block_dur * eval_rate
print(f"\n  Block duration: {block_dur}s -> {block_samples} samples at {eval_rate}Hz")
print(f"  p/T ratio: {total_cols}/{block_samples} = {total_cols/block_samples:.3f}")

# 8. What temporal resolution gives us
print("\n=== 8. Temporal sensitivity ===")
# How many blocks, and what fraction have coupling ON?
n_blocks = int(p1_bl.shape[0] / rate / block_dur)
if isinstance(bl_gate, np.ndarray):
    blocks_with_coupling = 0
    for b in range(n_blocks):
        s = int(b * block_dur * rate)
        e = int((b + 1) * block_dur * rate)
        if bl_gate[s:e].mean() > 0.1:
            blocks_with_coupling += 1
    print(f"  Total blocks: {n_blocks}")
    print(f"  Blocks with >10% coupling: {blocks_with_coupling}")
    print(f"  Block coupling rate: {blocks_with_coupling/n_blocks:.2%}")

print("\n=== Done ===")
