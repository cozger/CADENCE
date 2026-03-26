"""Kim filter on raw 52 blendshape AUs (no PCA) with sparse AU coupling."""
import numpy as np
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import find_valid_window, build_semisynthetic_base, generate_coupling_gate
from cadence.constants import MODALITY_SPECS_V2, BLENDSHAPE_SEGMENT_MAP
from cadence.basis.raised_cosine import raised_cosine_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.coupling.pathways import get_pathway_category
from cadence.significance.kim_filter import _kim_filter_single_channel, kim_filter_multivariate
import torch

cfg = load_config('configs/default.yaml')
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = [(n, load_session_from_cache(p, cfg)) for n, p in session_entries]
sess_list = [(n, s) for n, s in sess_list if s is not None]

s1, s2 = sess_list[-1][1], sess_list[0][1]
window = find_valid_window(s1, min_duration=1800)
base = build_semisynthetic_base(s1, s2, window[0], window[1])

# --- Extract RAW blendshapes (pre-PCA, 52 AUs) ---
# The session cache stores blendshapes_v2 (31 PCA+deriv+activity)
# but the raw AUs are in the original blendshapes key
# Check what's available
print("Available keys:", [k for k in base.keys() if 'blend' in k.lower()], flush=True)

# The raw blendshapes (52 AUs) should be accessible from the original session
# Let's check the original sessions
raw_key = 'p1_blendshapes'  # V1 key for raw 52+1 AUs
if raw_key not in s1:
    # Try extracting from PCA inverse... but we don't have loadings in base
    # Instead, use the p1/p2 blendshapes from the ORIGINAL sessions (pre-cross-dyad)
    print("Raw blendshapes not in base session. Checking original sessions...", flush=True)
    print("  s1 keys with 'blend':", [k for k in s1.keys() if 'blend' in k.lower()], flush=True)

# The cross-dyad base mixes P1 from s1, P2 from s2
# Raw AUs should be at 'p1_blendshapes' (53ch: 52 AU + activity) or similar
# Let's just work with whatever we have
bl_key = None
for k in ['p1_blendshapes', 'p1_blendshapes_raw']:
    if k in base:
        bl_key = k
        break

if bl_key is None:
    # Raw AUs not in cross-dyad base. Extract directly from original sessions.
    # P1 = s1 (y_32), P2 = s2 (y01) — same cross-dyad pairing
    print("Extracting raw AUs from original sessions...", flush=True)

    p1_raw = s1.get('p1_blendshapes', s1.get('p2_blendshapes'))
    p2_raw = s2.get('p1_blendshapes', s2.get('p2_blendshapes'))
    p1_raw_ts = s1.get('p1_blendshapes_ts', s1.get('p2_blendshapes_ts'))
    p2_raw_ts = s2.get('p1_blendshapes_ts', s2.get('p2_blendshapes_ts'))

    if p1_raw is None or p2_raw is None:
        # Try V2 keys — the blendshapes_v2 is PCA, not raw
        # Fall back to generating synthetic raw AUs from noise
        print("No raw AUs found. Generating synthetic 52-AU signals.", flush=True)
        fs = 30.0
        T_total = int(1800 * fs)
        rng = np.random.RandomState(42)

        # Realistic: AR(1) with rho=0.95 per AU
        p1_raw = np.zeros((T_total, 52), dtype=np.float32)
        p2_raw = np.zeros((T_total, 52), dtype=np.float32)
        for ch in range(52):
            p1_raw[0, ch] = rng.randn()
            p2_raw[0, ch] = rng.randn()
            for t in range(1, T_total):
                p1_raw[t, ch] = 0.95 * p1_raw[t-1, ch] + 0.31 * rng.randn()
                p2_raw[t, ch] = 0.95 * p2_raw[t-1, ch] + 0.31 * rng.randn()

        p1_raw_ts = np.arange(T_total) / fs
        p2_raw_ts = np.arange(T_total) / fs
    else:
        print(f"  P1 raw: {p1_raw.shape}, P2 raw: {p2_raw.shape}", flush=True)
else:
    p1_raw = base[bl_key]
    p2_raw = base[bl_key.replace('p1', 'p2')]
    p1_raw_ts = base[bl_key + '_ts']
    p2_raw_ts = base[bl_key.replace('p1', 'p2') + '_ts']

# --- Align to common native grid ---
fs = 30.0
t_s = max(float(p1_raw_ts[0]), float(p2_raw_ts[0]))
t_e = min(float(p1_raw_ts[-1]), float(p2_raw_ts[-1]))
t_e = min(t_e, t_s + 1800)  # cap at 1800s
T = int((t_e - t_s) * fs)
times = np.linspace(t_s, t_e, T)

C_au = min(p1_raw.shape[1], 52)  # 52 AUs (exclude activity)
src = np.stack([np.interp(times, p1_raw_ts, p1_raw[:, c]) for c in range(C_au)]).T
tgt = np.stack([np.interp(times, p2_raw_ts, p2_raw[:, c]) for c in range(C_au)]).T

# Normalize
for c in range(C_au):
    src[:, c] = (src[:, c] - src[:, c].mean()) / max(src[:, c].std(), 1e-8)
    tgt[:, c] = (tgt[:, c] - tgt[:, c].mean()) / max(tgt[:, c].std(), 1e-8)

print(f"\nRaw AUs: C={C_au}, T={T}, fs={fs} Hz", flush=True)

# --- Inject sparse AU coupling ---
# Mimic: mouth AUs move together (jaw + lips)
# From BLENDSHAPE_SEGMENT_MAP: mouth_affect [27,28,29,30,43,44], jaw [22,23,24,25]
coupled_aus = [22, 23, 27, 28, 29]  # jaw + mouth_affect subset
n_coupled = len(coupled_aus)
kappa = 0.4
duty_cycle = 0.10
lag_samples = int(2.0 * fs)

gate = generate_coupling_gate(T, fs, {
    'duty_cycle': duty_cycle, 'event_range_s': (3, 15), 'ramp_s': 1.0
}, seed=42)
gate_mask = gate > 0.5

# Inject coupling into selected AUs
tgt_coupled = tgt.copy()
for ch in coupled_aus:
    p1_lagged = np.roll(src[:, ch], lag_samples)
    p1_lagged[:lag_samples] = 0
    alpha = kappa * gate
    tgt_coupled[:, ch] = alpha * p1_lagged + np.sqrt(np.maximum(1 - alpha**2, 0)) * tgt[:, ch]

# Renormalize coupled channels
for ch in coupled_aus:
    tgt_coupled[:, ch] = (tgt_coupled[:, ch] - tgt_coupled[:, ch].mean()) / max(tgt_coupled[:, ch].std(), 1e-8)

print(f"Coupled AUs: {coupled_aus} ({n_coupled}/{C_au})", flush=True)
print(f"Gate duty: {gate_mask.mean():.1%}", flush=True)

# --- Basis convolution ---
category = get_pathway_category('blendshapes_v2', 'blendshapes_v2', cfg)
pw = cfg.get('pathway_temporal', {}).get(category, {'max_lag_seconds': 5.0, 'n_basis': 6})
basis, _ = raised_cosine_basis(n_basis=pw['n_basis'], max_lag_s=pw['max_lag_seconds'],
                                min_lag_s=0.0, sample_rate=fs, log_spacing=True)
dm = DesignMatrixBuilder(basis, ar_order=0, device='cuda')
conv, _ = dm.convolve_source(torch.tensor(src, dtype=torch.float32, device='cuda'),
                              np.ones(T, dtype=bool))
X_src = conv.view(T, C_au, pw['n_basis']).permute(1, 0, 2).cpu().numpy()

# --- Run Kim filter: coupled and null ---
Q, p_stay, em, penalty = 1e-6, 0.90, 5, 0.0
ar_order = 3

print(f"\nRunning MULTIVARIATE Kim filter (shared regime, penalty={penalty})...", flush=True)

t0 = time.perf_counter()
# Coupled
post_coupled, params_coupled = kim_filter_multivariate(
    tgt_coupled.T, X_src, ar_order=ar_order,
    p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
    Q_coeff=Q, em_iterations=em, complexity_penalty=penalty)
# Null
post_null, params_null = kim_filter_multivariate(
    tgt.T, X_src, ar_order=ar_order,
    p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
    Q_coeff=Q, em_iterations=em, complexity_penalty=penalty)
print(f"Done in {time.perf_counter()-t0:.1f}s\n", flush=True)

frac_coupled = float((post_coupled > 0.5).mean())
frac_null = float((post_null > 0.5).mean())

# Hit/FA against gate
hit = float((gate_mask & (post_coupled > 0.5)).sum() / max(gate_mask.sum(), 1))
fa = float((~gate_mask & (post_coupled > 0.5)).sum() / max((~gate_mask).sum(), 1))

# IoU
intersection = ((gate_mask) & (post_coupled > 0.5)).sum()
union = ((gate_mask) | (post_coupled > 0.5)).sum()
iou = float(intersection / max(union, 1))

print(f"=== MULTIVARIATE Kim results ===", flush=True)
print(f"Coupled session: frac={frac_coupled:.1%}  hit={hit:.1%}  fa={fa:.1%}  IoU={iou:.1%}", flush=True)
print(f"Null session:    frac={frac_null:.1%}", flush=True)
print(f"Discrimination:  {frac_coupled - frac_null:.1%}", flush=True)
print(f"", flush=True)
print(f"Per-channel sigma2_ratio:", flush=True)
sig2r_c = params_coupled['sigma2_ratio']
sig2r_n = params_null['sigma2_ratio']
for c in sorted(set(coupled_aus) | {c for c in range(C_au) if c in range(20, 35)}):
    label = "COUPLED" if c in coupled_aus else "null"
    print(f"  AU{c:>2}: coupled_s2r={sig2r_c[c]:.3f} null_s2r={sig2r_n[c]:.3f} {label}",
          flush=True)
