"""Corpus-level NMF expression discovery across all sessions.

Loads raw [0,1] blendshapes from all XDF files, concatenates across
all participants and full session durations, runs joint NMF.
The resulting components are stable, cross-dyad expression types
that can be converted to new composites for bl_coupling.py.

Optimizations:
  - Caches resampled signals to .npz (skip XDF parsing on re-runs)
  - Parallel k sweep via joblib (each k is independent)
  - Precomputes ||X||_F once for all k values
"""
import sys
sys.path.insert(0, 'C:/Users/optilab/desktop/CADENCE')

import numpy as np
import pyxdf
import glob
import os
import json
import time
from joblib import Parallel, delayed
from sklearn.decomposition import NMF

from cadence.significance.hawkes_coupling import MP_BLENDSHAPE_NAMES

FS = 30.0
RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
CACHE_PATH = 'C:/Users/optilab/Desktop/CADENCE/results/corpus_bl_raw_cache.npz'


def load_full_session_bl(xdf_path):
    """Load raw [0,1] blendshapes for full session.

    Returns list of (filename, person_label, signal) tuples.
    """
    fname = os.path.basename(xdf_path)
    try:
        data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)
    except Exception as e:
        print(f"  {fname}: Failed to load: {e}", flush=True)
        return []

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

    results = []
    for person in ['P1', 'P2']:
        if person not in landmarks:
            continue
        ts, d = landmarks[person]
        if len(ts) < 100:
            continue

        dur = ts[-1] - ts[0]
        T = int(dur * FS)
        if T < 300:
            continue

        t_grid = np.linspace(0, dur, T)
        ts_local = ts - ts[0]
        sig = np.empty((T, 52), dtype=np.float32)
        for c in range(52):
            sig[:, c] = np.interp(t_grid, ts_local, d[:, c])

        np.clip(sig, 0, 1, out=sig)
        results.append((fname, person, sig))
        print(f"  {fname} {person}: {T} samples ({dur:.0f}s)", flush=True)

    return results


def _fit_one_k(X, k, total_norm):
    """Fit NMF for one k value. Returns (k, explained, H, recon_err)."""
    nmf = NMF(n_components=k, init='nndsvda', max_iter=500, random_state=42)
    W = nmf.fit_transform(X)
    explained = 1 - (nmf.reconstruction_err_ / total_norm)
    return k, explained, nmf.components_, W


# ── Load or cache ─────────────────────────────────────────────────────

if os.path.exists(CACHE_PATH):
    print(f"Loading cached signals from {CACHE_PATH}", flush=True)
    t0 = time.time()
    cache = np.load(CACHE_PATH, allow_pickle=True)
    all_signals = list(cache['signals'])
    session_info = json.loads(str(cache['session_info']))
    load_time = time.time() - t0
    print(f"Loaded {len(all_signals)} recordings from cache in {load_time:.1f}s", flush=True)
else:
    xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
    print(f"Found {len(xdf_files)} XDF files in {RAW_DIR}\n", flush=True)

    t0 = time.time()
    all_signals = []
    session_info = []
    for xdf_path in xdf_files:
        for fname, person, sig in load_full_session_bl(xdf_path):
            all_signals.append(sig)
            session_info.append({'file': fname, 'person': person,
                                 'n_samples': sig.shape[0],
                                 'duration_s': sig.shape[0] / FS})
    load_time = time.time() - t0

    # Cache for future runs
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH,
                        signals=np.array(all_signals, dtype=object),
                        session_info=json.dumps(session_info))
    print(f"\nCached to {CACHE_PATH}", flush=True)

print(f"\nLoaded in {load_time:.1f}s", flush=True)
print(f"Total: {len(all_signals)} participant recordings", flush=True)
total_samples = sum(s.shape[0] for s in all_signals)
total_dur = total_samples / FS
print(f"Total samples: {total_samples:,} ({total_dur:.0f}s = {total_dur/60:.1f} min)", flush=True)

# ── Concatenate ───────────────────────────────────────────────────────

X = np.vstack(all_signals).astype(np.float64)
np.maximum(X, 0, out=X)
print(f"\nJoint matrix: {X.shape}", flush=True)
print(f"Non-zero fraction: {(X > 0.01).mean():.3f}", flush=True)

# Precompute Frobenius norm once (used by all k values)
total_norm = np.linalg.norm(X, 'fro')

# ── Parallel k sweep ─────────────────────────────────────────────────

k_values = [4, 5, 6, 7, 8, 10, 12]
print(f"\nRunning NMF for k={k_values} in parallel...", flush=True)
t0 = time.time()

results = Parallel(n_jobs=min(len(k_values), 8))(
    delayed(_fit_one_k)(X, k, total_norm) for k in k_values)

nmf_time = time.time() - t0
print(f"NMF sweep completed in {nmf_time:.1f}s\n", flush=True)

print("Reconstruction quality vs k:")
results.sort(key=lambda x: x[0])
for k, explained, H, W in results:
    print(f"  k={k:2d}: {explained:.1%} explained", flush=True)

# ── Detailed analysis at k=6 and k=8 ─────────────────────────────────

for target_k in [6, 8]:
    k, explained, H, W = [r for r in results if r[0] == target_k][0]

    print(f"\n{'='*70}", flush=True)
    print(f"NMF k={k} — Detailed Component Analysis ({explained:.1%} explained)", flush=True)
    print(f"{'='*70}", flush=True)

    for comp in range(k):
        loadings = H[comp]
        top_idx = np.argsort(loadings)[::-1][:8]

        w = W[:, comp]
        active_frac = (w > np.percentile(w, 75)).mean()

        top_aus = [(int(idx), MP_BLENDSHAPE_NAMES[idx], float(loadings[idx]))
                   for idx in top_idx if loadings[idx] > 0.1]

        name = '+'.join(MP_BLENDSHAPE_NAMES[idx] for idx in top_idx[:2])

        print(f"\n  Component {comp}: {name}", flush=True)
        print(f"    Activation: mean={w.mean():.4f}, std={w.std():.4f}, "
              f"max={w.max():.3f}, active>75pct: {active_frac:.1%}", flush=True)
        print(f"    Top AUs:", flush=True)
        for idx, au_name, loading in top_aus:
            print(f"      [{idx:2d}] {au_name:25s} = {loading:.3f}", flush=True)

        max_loading = loadings.max()
        composite_aus = [int(idx) for idx in range(52)
                         if loadings[idx] > 0.3 * max_loading]
        composite_names = [MP_BLENDSHAPE_NAMES[a] for a in composite_aus]
        print(f"    Suggested composite (>30% of max): {composite_aus}", flush=True)
        print(f"      AUs: {', '.join(composite_names)}", flush=True)

    # Save results
    out = {
        'k': k,
        'explained_variance': round(float(explained), 4),
        'n_sessions': len(set(s['file'] for s in session_info)),
        'n_participants': len(all_signals),
        'total_samples': int(total_samples),
        'total_duration_s': round(float(total_dur), 1),
        'nmf_time_s': round(nmf_time, 1),
        'components': [],
    }

    for comp in range(k):
        loadings = H[comp]
        top_idx = np.argsort(loadings)[::-1][:8]
        max_loading = loadings.max()
        composite_aus = [int(idx) for idx in range(52)
                         if loadings[idx] > 0.3 * max_loading]

        out['components'].append({
            'index': comp,
            'name': '+'.join(MP_BLENDSHAPE_NAMES[idx] for idx in top_idx[:2]),
            'top_aus': [(int(idx), MP_BLENDSHAPE_NAMES[idx], round(float(loadings[idx]), 4))
                        for idx in top_idx[:5]],
            'suggested_composite_aus': composite_aus,
            'suggested_composite_names': [MP_BLENDSHAPE_NAMES[a] for a in composite_aus],
            'mean_activation': round(float(W[:, comp].mean()), 5),
            'loadings': [round(float(x), 4) for x in loadings],
        })

    out_path = f'results/corpus_nmf_k{k}.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}", flush=True)

print(f"\nDone. Total wall time: {time.time() - t0:.1f}s", flush=True)
