"""Head-to-head: concordance vs envelope correlation vs ImCoh.
All three bands (theta, alpha, beta). Conversation vs baseline across all sessions."""

import sys, os, glob, numpy as np
from scipy.stats import wilcoxon
from scipy.signal import hilbert
from joblib import Parallel, delayed
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.significance.fast_cycles import _fft_bandpass, EEG_BANDS
from cadence.preprocess.eeg.coherence import eeg_band_coherence, DEFAULT_EEG_BANDS
from cadence.config import load_config
from cadence.data import discover_cached_sessions, load_session_from_cache
from scripts.run_session_v6 import load_xdf_session

CONDITION_ORDER = ['base_EO', 'base_EC', 'baseline', 'conv_1', 'PE', 'PE_1', 'PE_2',
                   'meditate_B', 'meditate_K', 'conv_2']
RAW_DIR = 'C:/Users/optilab/Desktop/CADENCE/raw sessions'
FS_OUT = 2.0


def process_one(session_name, config):
    xdf_files = glob.glob(os.path.join(RAW_DIR, f'{session_name}*.xdf'))
    if not xdf_files:
        return None
    try:
        sd = load_xdf_session(xdf_files[0])
    except Exception:
        return None
    markers = sd['markers']

    cached_sessions = discover_cached_sessions(config['session_cache'])
    cache_matches = [p for n, p in cached_sessions if session_name.lower() in n.lower()]
    if not cache_matches:
        return None
    cached = load_session_from_cache(cache_matches[0], config)

    if 'p1_eeg' not in cached or 'p2_eeg' not in cached:
        return None

    p1_eeg = cached['p1_eeg'].astype(np.float64)
    p2_eeg = cached['p2_eeg'].astype(np.float64)
    p1_ts = cached['p1_eeg_ts']
    lsl_offset = float(sd['landmarks']['P1'][0][0]) - float(
        cached.get('p1_blendshapes_ts', sd['landmarks']['P1'][0])[0])

    n_ch = min(14, p1_eeg.shape[1], p2_eeg.shape[1])
    p1_eeg = p1_eeg[:, :n_ch]
    p2_eeg = p2_eeg[:, :n_ch]
    mlen = min(len(p1_eeg), len(p2_eeg))
    p1_eeg = p1_eeg[:mlen]
    p2_eeg = p2_eeg[:mlen]
    p1_eeg -= p1_eeg.mean(axis=1, keepdims=True)
    p2_eeg -= p2_eeg.mean(axis=1, keepdims=True)
    for ch in range(n_ch):
        for arr in [p1_eeg, p2_eeg]:
            sd_val = arr[:, ch].std()
            if sd_val > 1e-8:
                arr[:, ch] = (arr[:, ch] - arr[:, ch].mean()) / sd_val
    fs_eeg = len(p1_ts) / (p1_ts[-1] - p1_ts[0])

    segments = []
    for seg_name in CONDITION_ORDER:
        t_s = markers.get(f'{seg_name}_start')
        t_e = markers.get(f'{seg_name}_stop')
        if t_s is not None and t_e is not None and t_e > t_s:
            segments.append((seg_name, t_s, t_e))
    if not segments:
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dur = mlen / fs_eeg
    t_grid = np.arange(0, dur, 1.0 / FS_OUT)
    t_grid_lsl = t_grid + (p1_ts[0] + lsl_offset)
    N = len(t_grid)

    both = np.vstack([p1_eeg.T, p2_eeg.T])
    both_t = torch.as_tensor(both, dtype=torch.float32, device=device)

    cond_results = {}

    for band_name, (lo, hi) in EEG_BANDS.items():
        filt = _fft_bandpass(both_t, fs_eeg, (lo, hi), device).cpu().numpy()
        p1_filt = filt[:n_ch].T
        p2_filt = filt[n_ch:].T

        p1_env = np.abs(hilbert(p1_filt, axis=0)).mean(axis=1)
        p2_env = np.abs(hilbert(p2_filt, axis=0)).mean(axis=1)

        t_eeg = np.arange(mlen) / fs_eeg
        p1_2hz = np.interp(t_grid, t_eeg, p1_env)
        p2_2hz = np.interp(t_grid, t_eeg, p2_env)

        for arr in [p1_2hz, p2_2hz]:
            mu, sd_v = arr.mean(), arr.std()
            if sd_v > 1e-8:
                arr[:] = (arr - mu) / sd_v
            else:
                arr[:] = 0.0

        conc = (p1_2hz + p2_2hz) / 2.0

        # Envelope correlation in 2s sliding windows
        win_samples = int(2.0 * FS_OUT)
        env_corr = np.zeros(N)
        for t_i in range(N):
            w_start = max(0, t_i - win_samples // 2)
            w_end = min(N, t_i + win_samples // 2)
            if w_end - w_start < 3:
                continue
            s1 = p1_2hz[w_start:w_end]
            s2 = p2_2hz[w_start:w_end]
            sd1, sd2 = s1.std(), s2.std()
            if sd1 > 1e-8 and sd2 > 1e-8:
                env_corr[t_i] = np.corrcoef(s1, s2)[0, 1]

        for seg_name, t0, t1 in segments:
            mask = (t_grid_lsl >= t0) & (t_grid_lsl <= t1)
            if mask.sum() < 3:
                continue
            if seg_name not in cond_results:
                cond_results[seg_name] = {}
            cond_results[seg_name][f'conc_{band_name}'] = float(conc[mask].mean())
            cond_results[seg_name][f'envcorr_{band_name}'] = float(env_corr[mask].mean())

    # ImCoh
    try:
        _, coh_imag, coh_times, _ = eeg_band_coherence(
            p1_eeg, p2_eeg, fs=fs_eeg, use_imcoh=True)
        coh_times_lsl = coh_times + (p1_ts[0] + lsl_offset)
        imag_avg = coh_imag.mean(axis=1)
        band_names_coh = list(DEFAULT_EEG_BANDS.keys())
        for seg_name, t0, t1 in segments:
            mask = (coh_times_lsl >= t0) & (coh_times_lsl <= t1)
            if mask.sum() < 3:
                continue
            if seg_name not in cond_results:
                cond_results[seg_name] = {}
            for bi, bn in enumerate(band_names_coh):
                if bi < imag_avg.shape[0]:
                    cond_results[seg_name][f'imcoh_{bn}'] = float(imag_avg[bi, mask].mean())
    except Exception:
        pass

    return {'session': session_name, 'conditions': cond_results}


def main():
    config = load_config()
    xdf_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.xdf')))
    sessions = [os.path.splitext(os.path.basename(f))[0].lower() for f in xdf_files]
    print(f'Processing {len(sessions)} sessions...')
    results = Parallel(n_jobs=-1)(
        delayed(process_one)(s, config) for s in sessions)
    results = [r for r in results if r is not None]
    print(f'Got {len(results)} sessions\n')

    features = []
    for bn in ['theta', 'alpha', 'beta']:
        features.extend([f'conc_{bn}', f'envcorr_{bn}', f'imcoh_{bn}'])

    print('CONCORDANCE vs ENVELOPE CORRELATION vs IMAGINARY COHERENCE')
    print('All three bands (theta, alpha, beta) -- Conversation vs Baseline')
    print('=' * 85)
    print(f'  {"Feature":>18s} | conv_mean | base_mean | diff      | p-value | sig  | n')
    print('  ' + '-' * 80)

    for f in features:
        conv_vals = []
        base_vals = []
        for r in results:
            cv = []
            bv = []
            for cname, cr in r['conditions'].items():
                v = cr.get(f)
                if v is None:
                    continue
                if cname.startswith('conv'):
                    cv.append(v)
                elif cname.startswith('base') or cname == 'baseline':
                    bv.append(v)
            if cv and bv:
                conv_vals.append(np.mean(cv))
                base_vals.append(np.mean(bv))

        if len(conv_vals) >= 5:
            diff = np.mean(conv_vals) - np.mean(base_vals)
            stat, p = wilcoxon(conv_vals, base_vals)
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            print(f'  {f:>18s} | {np.mean(conv_vals):+9.4f} | '
                  f'{np.mean(base_vals):+9.4f} | {diff:+9.4f} | '
                  f'{p:.4f}  | {sig:4s} | {len(conv_vals)}')
        else:
            print(f'  {f:>18s} | insufficient (n={len(conv_vals)})')

    # Summary
    print('\n  SUMMARY BY METRIC TYPE (best p across bands):')
    for metric in ['conc', 'envcorr', 'imcoh']:
        best_p = 1.0
        best_band = ''
        for bn in ['theta', 'alpha', 'beta']:
            f = f'{metric}_{bn}'
            conv_vals = []
            base_vals = []
            for r in results:
                cv = []
                bv = []
                for cname, cr in r['conditions'].items():
                    v = cr.get(f)
                    if v is None:
                        continue
                    if cname.startswith('conv'):
                        cv.append(v)
                    elif cname.startswith('base') or cname == 'baseline':
                        bv.append(v)
                if cv and bv:
                    conv_vals.append(np.mean(cv))
                    base_vals.append(np.mean(bv))
            if len(conv_vals) >= 5:
                _, p = wilcoxon(conv_vals, base_vals)
                if p < best_p:
                    best_p = p
                    best_band = bn
        sig = '***' if best_p < 0.01 else ('**' if best_p < 0.05 else ('*' if best_p < 0.1 else ''))
        print(f'    {metric:>8s}: best p={best_p:.4f} ({best_band}) {sig}')


if __name__ == '__main__':
    main()
