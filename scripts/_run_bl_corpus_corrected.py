"""Corpus-level BL event analysis with corrected AU indices.

Runs bl_two_stage_coupling on all sessions with:
  1. Corrected hand-defined composites (MediaPipe ordering)
  2. NMF-discovered composites from corpus analysis
  3. Combined set
Reports per-composite co-occurrence rates, p-values, and Fisher aggregation.
"""
import sys, os, time, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyxdf
from joblib import Parallel, delayed
from scipy.stats import chi2 as chi2_dist

from cadence.significance.bl_coupling import bl_two_stage_coupling, EXPRESSION_COMPOSITES
from cadence.ingest.roles import roles_dict_from_session as _detect_roles

FS = 30.0
RAW_DIR = r'C:\Users\optilab\Desktop\CADENCE\raw sessions'

# Load NMF composites
with open('results/corpus_nmf_k6.json') as f:
    nmf_data = json.load(f)

NMF_COMPOSITES = {}
for comp in nmf_data['components']:
    aus = tuple(comp['suggested_composite_aus'])
    top_names = [a[1] for a in comp['top_aus'][:2]]
    # Skip pure eye components
    if all('eye' in n.lower() or 'blink' in n.lower() or 'Look' in n for n in top_names):
        continue
    short_name = comp['name'].split('+')[0][:12]
    NMF_COMPOSITES[f'nmf_{short_name}'] = aus

print("Hand-defined composites (corrected MediaPipe indices):")
for name, aus in EXPRESSION_COMPOSITES.items():
    print(f"  {name}: {aus}")
print(f"\nNMF composites:")
for name, aus in NMF_COMPOSITES.items():
    print(f"  {name}: {aus}")

COMBINED = dict(EXPRESSION_COMPOSITES)
COMBINED.update(NMF_COMPOSITES)


def extract_conversations(xdf_path):
    """Extract conv segments with role detection."""
    session_name = os.path.splitext(os.path.basename(xdf_path))[0]
    try:
        data, _ = pyxdf.load_xdf(xdf_path)
    except Exception as e:
        print(f"  ERROR loading {session_name}: {e}")
        return []

    roles = _detect_roles(data)
    p1_role = roles.get('p1_role', 'unknown')
    p2_role = roles.get('p2_role', 'unknown')

    marker_times = {}
    for stream in data:
        if stream['info']['type'][0] == 'Markers':
            for t, v in zip(stream['time_stamps'], stream['time_series']):
                marker_times[v[0]] = t

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
        print(f"  {session_name}: missing landmarks")
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
            dur = t_end - t_start
            T = int(dur * FS)
            t_grid = np.linspace(0, dur, T)

            sigs = {}
            for p in ['P1', 'P2']:
                ts, d = landmarks[p]
                m = (ts >= t_start) & (ts <= t_end)
                if m.sum() < 100:
                    continue
                sigs[p] = np.stack([np.interp(t_grid, ts[m] - t_start, d[m, c])
                                    for c in range(52)], axis=1)

            if 'P1' not in sigs or 'P2' not in sigs:
                continue

            results.append((session_name, conv, sigs['P1'], sigs['P2'],
                            T, dur, p1_role, p2_role))
        except Exception as e:
            print(f"  {session_name}/{conv}: ERROR {e}")

    return results


def run_segment(session_name, conv, p1, p2, T, dur, p1_role, p2_role):
    """Run pipeline with combined composites, both directions."""
    outputs = []
    for src_label, tgt_label, src, tgt in [('P1', 'P2', p1, p2),
                                             ('P2', 'P1', p2, p1)]:
        src_role = p1_role if src_label == 'P1' else p2_role
        tgt_role = p1_role if tgt_label == 'P1' else p2_role
        role_dir = f"{src_role}->{tgt_role}"

        try:
            res = bl_two_stage_coupling(
                src, tgt, FS,
                composites=COMBINED,
                max_lag_s=5.0, lag_step_s=0.1, smooth_s=3.0,
                n_surrogates=100, target_fa=0.05,
                event_prominence=0.3, min_event_iei_s=3.0,
                n_surrogates_event=500,
                population_prior=(2.5, 1.0), seed=42)

            for comp_name, cat in res.catalogs.items():
                outputs.append({
                    'session': session_name, 'conv': conv,
                    'role_dir': role_dir,
                    'composite': comp_name,
                    'n_src': cat.n_events_a,
                    'n_tgt': cat.n_events_b,
                    'n_cooc': cat.n_cooccurrences,
                    'cooc_rate': cat.cooccurrence_rate,
                    'null_rate': cat.null_cooccurrence_rate,
                    'p_value': cat.session_p_value,
                    'mean_lag': cat.mean_lag,
                    'n_mimicry': cat.n_mimicry,
                    'n_shared': cat.n_shared_stimulus,
                    'n_coincidence': cat.n_coincidence,
                    'n_a_led': cat.n_a_led,
                    'n_b_led': cat.n_b_led,
                    'coupling_frac': float(res.mask_continuous.mean()),
                    'z_max': float(res.z_continuous.max()),
                    'est_lag': res.estimated_lag_s,
                })
        except Exception as e:
            print(f"  {session_name}/{conv} {role_dir}: ERROR {e}")

    return outputs


# ── Run ───────────────────────────────────────────────────────────────

xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
print(f"\nFound {len(xdf_files)} XDF files")

print("Extracting segments (threaded I/O)...", flush=True)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=min(len(xdf_files), 4)) as pool:
    results_io = list(pool.map(extract_conversations, xdf_files))
all_segments = [seg for batch in results_io for seg in batch]

print(f"\nTotal segments: {len(all_segments)}")
for seg in all_segments:
    print(f"  {seg[0]:>20s} {seg[1]:>6s} {seg[5]:.0f}s P1={seg[6]} P2={seg[7]}")

print(f"\nRunning pipeline...", flush=True)
t0 = time.perf_counter()
results_nested = Parallel(n_jobs=-1)(
    delayed(run_segment)(*seg) for seg in all_segments)
elapsed = time.perf_counter() - t0

all_results = [r for batch in results_nested for r in batch]
print(f"\nCompleted in {elapsed:.1f}s, {len(all_results)} entries\n")

# ── Per-segment summary ──────────────────────────────────────────────

print(f"{'session':>15} {'conv':>6} {'dir':>15} {'comp':>15} "
      f"{'src':>4} {'tgt':>4} {'cooc':>4} {'p':>7} {'mim':>3} {'shr':>3} {'lag':>5}")
print("-" * 95)

for r in sorted(all_results, key=lambda x: (x['session'], x['conv'], x['role_dir'], x['p_value'])):
    if r['n_src'] == 0 and r['n_tgt'] == 0:
        continue
    sig = '**' if r['p_value'] < 0.01 else ('*' if r['p_value'] < 0.05 else '')
    lag_s = f"{r['mean_lag']:.1f}" if r['mean_lag'] is not None else "—"
    print(f"{r['session']:>15} {r['conv']:>6} {r['role_dir']:>15} {r['composite']:>15} "
          f"{r['n_src']:4d} {r['n_tgt']:4d} {r['n_cooc']:4d} {r['p_value']:7.3f}{sig:>2} "
          f"{r['n_mimicry']:3d} {r['n_shared']:3d} {lag_s:>5}")

# ── Aggregation by composite ─────────────────────────────────────────

print(f"\n{'='*80}")
print("AGGREGATION BY COMPOSITE")
print(f"{'='*80}")

comp_names = sorted(set(r['composite'] for r in all_results))
for comp in comp_names:
    entries = [r for r in all_results if r['composite'] == comp
               and r['n_src'] > 0 and r['n_tgt'] > 0]
    if not entries:
        continue

    n = len(entries)
    rates = [e['cooc_rate'] for e in entries]
    nulls = [e['null_rate'] for e in entries]
    pvals = [e['p_value'] for e in entries]
    n_sig = sum(1 for p in pvals if p < 0.05)
    tot_cooc = sum(e['n_cooc'] for e in entries)
    tot_mim = sum(e['n_mimicry'] for e in entries)
    tot_shared = sum(e['n_shared'] for e in entries)
    tot_coinc = sum(e['n_coincidence'] for e in entries)
    lags = [e['mean_lag'] for e in entries if e['mean_lag'] is not None]

    # Fisher's method
    chi2_val = -2 * np.sum(np.log(np.maximum(pvals, 1e-10)))
    fisher_p = 1 - chi2_dist.cdf(chi2_val, 2 * n)
    fsig = '**' if fisher_p < 0.01 else ('*' if fisher_p < 0.05 else '')

    excess = np.mean(rates) - np.mean(nulls) if nulls else 0
    lag_str = f"lag={np.mean(lags):.1f}s" if lags else ""

    print(f"\n  {comp:>15}: n={n:2d}  rate={np.mean(rates):.0%} null={np.mean(nulls):.0%} "
          f"excess={excess:>+.0%}  sig={n_sig}/{n}  Fisher={fisher_p:.4f}{fsig}")
    print(f"                  cooc={tot_cooc} mimicry={tot_mim} shared={tot_shared} "
          f"coinc={tot_coinc} {lag_str}")

# Save
out_path = 'results/bl_corpus_corrected.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
