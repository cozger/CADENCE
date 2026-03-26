"""Kim filter parameter sweep on semisynthetic data.

Builds the semisynthetic base once, injects BL coupling, then sweeps
Kim parameters directly — bypasses Stage 1/1.5/cross-modal entirely.
"""
import numpy as np
import time
import itertools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed
from cadence.config import load_config
from cadence.data.alignment import discover_cached_sessions, load_session_from_cache
from cadence.synthetic import build_semisynthetic_base, inject_coupling_modality
from cadence.constants import MODALITY_SPECS_V2
from cadence.basis.raised_cosine import raised_cosine_basis
from cadence.basis.design_matrix import DesignMatrixBuilder
from cadence.significance.kim_filter import _kim_filter_single_channel
import torch

cfg = load_config('configs/default.yaml')

# --- Build semisynthetic session ---
print("Loading sessions...", flush=True)
session_entries = discover_cached_sessions(cfg['session_cache'])
sess_list = []
for name, path in session_entries:
    s = load_session_from_cache(path, cfg)
    if s is not None:
        sess_list.append((name, s))
        print(f"  {name} ({s.get('duration',0):.0f}s)", flush=True)

# Pick two sessions for cross-dyad pair
s1_name, s1 = sess_list[-1]  # y_32
s2_name, s2 = sess_list[0]   # y01
print(f"\nPair: P1={s1_name}, P2={s2_name}", flush=True)

from cadence.synthetic import find_valid_window
window = find_valid_window(s1, min_duration=1800)
if window is None:
    print("ERROR: No valid 1800s window for P1!", flush=True)
    sys.exit(1)
t_start_pair, t_end_pair = window
print(f"  Window: [{t_start_pair:.0f}s, {t_end_pair:.0f}s]", flush=True)

print("Building semisynthetic base...", flush=True)
base = build_semisynthetic_base(s1, s2, t_start_pair, t_end_pair)
print(f"  Base built, duration={base.get('duration',0):.0f}s", flush=True)

# Inject coupling
kappa = 0.4
duty_cycle = 0.10
target_mod = 'blendshapes_v2'
print(f"Injecting: kappa={kappa}, duty={duty_cycle:.0%}, mod={target_mod}", flush=True)
session = inject_coupling_modality(base, target_mod, kappa,
                                    duty_cycle=duty_cycle, seed=42)

# Get coupling gate for evaluation
gate = session.get('coupling_gates', {}).get(target_mod)
if gate is not None:
    fs_gate = MODALITY_SPECS_V2[target_mod][1]
    gate_mask = gate > 0.5
    print(f"Gate: duty={gate_mask.mean():.1%}, fs={fs_gate} Hz", flush=True)
else:
    print("WARNING: no coupling gate found!", flush=True)
    gate_mask = None

# --- Extract BL signals at native rate ---
p1_bl = session[f'p1_{target_mod}']
p2_bl = session[f'p2_{target_mod}']
p1_ts = session[f'p1_{target_mod}_ts']
p2_ts = session[f'p2_{target_mod}_ts']
fs_native = float(MODALITY_SPECS_V2[target_mod][1])

# Align to common time grid at native rate
t_start = max(float(p1_ts[0]), float(p2_ts[0]))
t_end = min(float(p1_ts[-1]), float(p2_ts[-1]))
T_native = int((t_end - t_start) * fs_native)
native_times = np.linspace(t_start, t_end, T_native)

# All channels (not just discovery-selected)
C_all = p1_bl.shape[1]
src_native = np.stack([np.interp(native_times, p1_ts, p1_bl[:, c])
                        for c in range(C_all)]).T
tgt_native = np.stack([np.interp(native_times, p2_ts, p2_bl[:, c])
                        for c in range(C_all)]).T

print(f"Signals: C={C_all}, T={T_native}, fs={fs_native} Hz", flush=True)

# --- Build basis at native rate ---
from cadence.coupling.pathways import get_pathway_category
category = get_pathway_category(target_mod, target_mod, cfg)
pw_temporal = cfg.get('pathway_temporal', {}).get(
    category, {'max_lag_seconds': 5.0, 'n_basis': 6})
n_basis = pw_temporal['n_basis']
max_lag = pw_temporal['max_lag_seconds']

native_basis, _ = raised_cosine_basis(
    n_basis=n_basis, max_lag_s=max_lag, min_lag_s=0.0,
    sample_rate=fs_native, log_spacing=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dm = DesignMatrixBuilder(native_basis, ar_order=0, device=device)
src_t = torch.tensor(src_native, dtype=torch.float32, device=device)
convolved, _ = dm.convolve_source(src_t, np.ones(T_native, dtype=bool))
X_src = convolved.view(T_native, C_all, n_basis).permute(
    1, 0, 2).cpu().numpy()  # (C, T, nb)

y_mc = tgt_native.T  # (C, T)

# Resample gate to native times for evaluation
if gate_mask is not None:
    gate_times = np.arange(len(gate)) / fs_gate
    gate_native = np.interp(native_times, gate_times, gate.astype(float)) > 0.5
else:
    gate_native = np.zeros(T_native, dtype=bool)

duty = gate_native.mean()
print(f"Native gate duty: {duty:.1%}", flush=True)

# Which channels are coupled?
from cadence.synthetic import _get_coupled_indices
coupled_idx = _get_coupled_indices(target_mod)
n_coupled = len(coupled_idx)
print(f"Coupled channels: {coupled_idx} ({n_coupled}/{C_all})", flush=True)

# --- Parameter sweep ---
Q_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
p_stay_values = [0.90, 0.95, 0.98, 0.99]
em_values = [3, 5]

configs = list(itertools.product(Q_values, p_stay_values, em_values))
n_configs = len(configs)
total_jobs = n_configs * C_all
print(f"\nSweeping {n_configs} configs x {C_all} channels = {total_jobs} jobs...",
      flush=True)

# Flatten: run ALL (config, channel) pairs in parallel
def run_single(Q, p_stay, em_iter, ch):
    post, params = _kim_filter_single_channel(
        y_mc[ch], X_src[ch], ar_order=3,
        p_stay_coupled=p_stay, p_stay_uncoupled=p_stay,
        Q_coeff=Q, em_iterations=em_iter)
    return (Q, p_stay, em_iter, ch, post, params)

t0 = time.perf_counter()
all_results = Parallel(n_jobs=-1, verbose=1)(
    delayed(run_single)(Q, p, e, ch)
    for Q, p, e in configs
    for ch in range(C_all))
elapsed = time.perf_counter() - t0
print(f"\nAll {total_jobs} jobs done in {elapsed:.1f}s", flush=True)

# Regroup by config
from collections import defaultdict
grouped = defaultdict(dict)
for Q, p_stay, em_iter, ch, post, params in all_results:
    grouped[(Q, p_stay, em_iter)][ch] = (post, params)

# Compute metrics per config
results = []
for (Q, p_stay, em_iter), ch_results in grouped.items():
    coupled_fracs = [(ch_results[c][0] > 0.5).mean() for c in coupled_idx]
    null_idx = [c for c in range(C_all) if c not in coupled_idx]
    null_fracs = [(ch_results[c][0] > 0.5).mean() for c in null_idx]

    best_c = max(coupled_idx, key=lambda c: (ch_results[c][0] > 0.5).mean())
    post_best = ch_results[best_c][0] > 0.5
    hit = float((gate_native & post_best).sum() / max(gate_native.sum(), 1))
    fa = float((~gate_native & post_best).sum() / max((~gate_native).sum(), 1))

    disc = np.mean(coupled_fracs) - np.mean(null_fracs)
    sig2r_coupled = np.mean([ch_results[c][1]['sigma2_ratio'] for c in coupled_idx])
    sig2r_null = np.mean([ch_results[c][1]['sigma2_ratio'] for c in null_idx])

    results.append({
        'Q': Q, 'p_stay': p_stay, 'em': em_iter,
        'coupled_mean': np.mean(coupled_fracs),
        'null_mean': np.mean(null_fracs),
        'disc': disc,
        'best_hit': hit, 'best_fa': fa,
        'sig2r_coupled': sig2r_coupled,
        'sig2r_null': sig2r_null,
    })

# Sort by discrimination
results.sort(key=lambda r: -r['disc'])

print(f"\n{'Q':>8} {'p_stay':>6} {'em':>3} | "
      f"{'coupled':>7} {'null':>7} {'disc':>6} | "
      f"{'hit':>5} {'fa':>5} | {'sig2r_c':>7} {'sig2r_n':>7}", flush=True)
print("-" * 85, flush=True)
for r in results[:20]:
    print(f"{r['Q']:>8.1e} {r['p_stay']:>6.2f} {r['em']:>3d} | "
          f"{r['coupled_mean']:>7.1%} {r['null_mean']:>7.1%} "
          f"{r['disc']:>6.1%} | "
          f"{r['best_hit']:>5.1%} {r['best_fa']:>5.1%} | "
          f"{r['sig2r_coupled']:>7.3f} {r['sig2r_null']:>7.3f}", flush=True)

good = [r for r in results if r['best_fa'] < 0.15]
if good:
    good.sort(key=lambda r: -r['best_hit'])
    print(f"\n=== Best by hit rate (FA<15%) ===", flush=True)
    for r in good[:5]:
        print(f"  Q={r['Q']:.1e} p_stay={r['p_stay']:.2f} em={r['em']}: "
              f"hit={r['best_hit']:.1%} fa={r['best_fa']:.1%} "
              f"coupled={r['coupled_mean']:.1%} null={r['null_mean']:.1%} "
              f"sig2r_c={r['sig2r_coupled']:.3f} sig2r_n={r['sig2r_null']:.3f}",
              flush=True)
