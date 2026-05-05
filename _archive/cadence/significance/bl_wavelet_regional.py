"""Per-region anatomical wavelet coherence on facial blendshapes.

Parallel module to bl_wavelet.py. Computes wavelet coherence per anatomical
region (mouth, eye, brow, cheek, nose) instead of via the legacy AFFECT_AUS
functional grouping. Designed for the standalone region-specificity validation
battery; does not feed into V11/V12 scaffolds (yet).

Reuses the existing CWT and surrogate-shifting machinery from bl_wavelet.py
(compute_au_cwt, BAND_*) — only the per-region pooling and z-scoring is new.

Two outputs per region:
  - AU-pooled wavelet coherence z-score (the primary coupling metric;
    preserves omnibus power across all AUs in the region).
  - PCA loadings on the regional AU panel (interpretation-only; tells you
    which AUs within a region drive the variance, not the coupling).

The PCA is decoupled from the coupling estimate by design: coherence uses all
AUs jointly, while loadings are a separate readout for human inspection.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

from cadence.constants import AU_REGIONS
from cadence.significance.bl_wavelet import (
    compute_au_cwt, surrogate_coherence_z,
    BAND_STATE, BAND_EXPRESSION, BAND_SPEECH,
)


# ── Per-region PCA (interpretation only) ───────────────────────────────

def compute_regional_pca(bl_data, regions=None, n_components=2):
    """Fit PCA per anatomical region for loadings inspection.

    Not used for coupling estimation — coupling uses AU-pooled coherence.
    PC1/PC2 loadings tell you which AUs within a region drive its variance.

    Args:
        bl_data: (T, 52+) blendshape array (raw or z-scored).
        regions: dict region_name -> list of AU indices. Default: AU_REGIONS.
        n_components: number of PCs per region (default 2: dominant + asymmetry).

    Returns:
        dict region_name -> {
            'pc_timeseries': (T, n_components) projection,
            'loadings':      (n_components, n_aus_in_region),
            'var_explained': (n_components,),
            'au_indices':    list of AU indices in region (column order in loadings),
            'mean':          (n_aus_in_region,) per-AU mean used for centering,
        }
    """
    if regions is None:
        regions = AU_REGIONS

    T, n_ch = bl_data.shape
    out = {}

    for region_name, au_idxs in regions.items():
        aus = [a for a in au_idxs if 0 <= a < n_ch]
        if len(aus) < 1:
            continue
        panel = bl_data[:, aus].astype(np.float64)  # (T, k)
        if panel.shape[1] == 0 or panel.std() < 1e-10:
            continue

        # Center each AU
        panel_mean = panel.mean(axis=0)
        X = panel - panel_mean[None, :]

        # SVD-based PCA (numerically stable, no sklearn dependency)
        # X = U S V^T, components are rows of V^T
        n_comp = min(n_components, X.shape[1])
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        loadings = Vt[:n_comp, :].astype(np.float32)             # (n_comp, k)
        pc_ts = (U[:, :n_comp] * S[:n_comp][None, :]).astype(np.float32)  # (T, n_comp)

        var_total = (S ** 2).sum()
        var_exp = ((S[:n_comp] ** 2) / max(var_total, 1e-12)).astype(np.float32)

        out[region_name] = {
            'pc_timeseries': pc_ts,
            'loadings': loadings,
            'var_explained': var_exp,
            'au_indices': list(aus),
            'mean': panel_mean.astype(np.float32),
        }

    return out


# ── Multi-region surrogate-z (single GPU pass) ─────────────────────────

def _multi_region_surrogate_z_gpu(w1, w2, region_au_lists, sigma_samples, shifts):
    """Compute surrogate-z wavelet coherence for multiple AU groups in one pass.

    Shares the surrogate shift loop across all regions for ~Nx speedup over
    calling surrogate_coherence_z N times. Within each shift iteration, reuses
    the shifted CWT coefficients to compute per-region cross-/auto-spectra.

    Args:
        w1, w2: (n_freqs, T, n_ch_full) complex CWT coefficients for both
            participants. Must contain coefficients for ALL AUs across all
            regions; per-region selection happens here.
        region_au_lists: dict region_name -> list of AU indices.
        sigma_samples: temporal smoothing kernel width (samples).
        shifts: list/array of integer circular-shift offsets for surrogates.

    Returns:
        dict region_name -> {
            'real_coh':  (n_freqs, T) float32,
            'null_mean': (n_freqs, T) float32,
            'null_std':  (n_freqs, T) float32,
        }
    """
    n_freqs, T, _ = w1.shape
    n_surr = len(shifts)

    c1_full = torch.tensor(w1, dtype=torch.complex64, device='cuda')
    c2_full = torch.tensor(w2, dtype=torch.complex64, device='cuda')

    # Smoothing kernel
    kernel_size = int(6 * sigma_samples) | 1
    t = torch.arange(kernel_size, device='cuda', dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    pad = kernel_size // 2

    def smooth_real(x):
        return torch.nn.functional.conv1d(x.unsqueeze(1), kernel, padding=pad).squeeze(1)

    def smooth_complex(x):
        r = torch.nn.functional.conv1d(x.real.unsqueeze(1), kernel, padding=pad).squeeze(1)
        i = torch.nn.functional.conv1d(x.imag.unsqueeze(1), kernel, padding=pad).squeeze(1)
        return torch.complex(r, i)

    # Pre-compute per-region selections + auto1 (P1's auto-spectrum is invariant
    # under surrogate shifts of P2)
    per_region = {}
    for region_name, aus in region_au_lists.items():
        c1_r = c1_full[:, :, aus]
        auto1_r = (c1_r.abs() ** 2).sum(dim=2)
        auto1_r_s = smooth_real(auto1_r)
        per_region[region_name] = {
            'aus': aus,
            'c1': c1_r,
            'auto1_s': auto1_r_s,
            'surr_sum': torch.zeros_like(auto1_r_s),
            'surr_sum_sq': torch.zeros_like(auto1_r_s),
        }

    # Real coherence per region
    real_coh = {}
    for region_name, info in per_region.items():
        c2_r = c2_full[:, :, info['aus']]
        cross = (info['c1'] * c2_r.conj()).sum(dim=2)
        auto2 = (c2_r.abs() ** 2).sum(dim=2)
        cross_s = smooth_complex(cross)
        auto2_s = smooth_real(auto2)
        coh = cross_s.abs() ** 2 / (info['auto1_s'] * auto2_s + 1e-10)
        real_coh[region_name] = coh

    # Surrogate loop — single shift roll feeds all regions
    for shift in shifts:
        c2_shifted_full = torch.roll(c2_full, int(shift), dims=1)
        for region_name, info in per_region.items():
            c2_r = c2_shifted_full[:, :, info['aus']]
            cross_s = smooth_complex((info['c1'] * c2_r.conj()).sum(dim=2))
            auto2_s = smooth_real((c2_r.abs() ** 2).sum(dim=2))
            surr_coh = cross_s.abs() ** 2 / (info['auto1_s'] * auto2_s + 1e-10)
            info['surr_sum'] += surr_coh
            info['surr_sum_sq'] += surr_coh ** 2

    out = {}
    for region_name, info in per_region.items():
        null_mean = info['surr_sum'] / n_surr
        null_std = torch.sqrt(info['surr_sum_sq'] / n_surr - null_mean ** 2 + 1e-10)
        rc = real_coh[region_name]
        out[region_name] = {
            'real_coh':  rc.cpu().numpy().astype(np.float32),
            'null_mean': null_mean.cpu().numpy().astype(np.float32),
            'null_std':  null_std.cpu().numpy().astype(np.float32),
        }

    return out


def _multi_region_surrogate_z_cpu(w1, w2, region_au_lists, sigma_samples, shifts):
    """CPU fallback for _multi_region_surrogate_z_gpu. Slower but no torch dep."""
    from scipy.ndimage import gaussian_filter1d

    n_surr = len(shifts)

    out = {}
    for region_name, aus in region_au_lists.items():
        c1 = w1[:, :, aus]
        c2 = w2[:, :, aus]

        cross = (c1 * np.conj(c2)).sum(axis=2)
        auto1 = (np.abs(c1) ** 2).sum(axis=2)
        auto2 = (np.abs(c2) ** 2).sum(axis=2)
        cross_s = gaussian_filter1d(cross, sigma=sigma_samples, axis=1)
        auto1_s = gaussian_filter1d(auto1, sigma=sigma_samples, axis=1)
        auto2_s = gaussian_filter1d(auto2, sigma=sigma_samples, axis=1)
        real_coh = (np.abs(cross_s) ** 2 / (auto1_s * auto2_s + 1e-10))

        surr_sum = np.zeros_like(real_coh, dtype=np.float64)
        surr_sum_sq = np.zeros_like(real_coh, dtype=np.float64)
        for shift in shifts:
            c2_sh = np.roll(c2, int(shift), axis=1)
            cross_sh = (c1 * np.conj(c2_sh)).sum(axis=2)
            auto2_sh = (np.abs(c2_sh) ** 2).sum(axis=2)
            cross_sh_s = gaussian_filter1d(cross_sh, sigma=sigma_samples, axis=1)
            auto2_sh_s = gaussian_filter1d(auto2_sh, sigma=sigma_samples, axis=1)
            surr_coh = (np.abs(cross_sh_s) ** 2 / (auto1_s * auto2_sh_s + 1e-10))
            surr_sum += surr_coh
            surr_sum_sq += surr_coh ** 2

        null_mean = surr_sum / n_surr
        null_std = np.sqrt(surr_sum_sq / n_surr - null_mean ** 2 + 1e-10)

        out[region_name] = {
            'real_coh':  real_coh.astype(np.float32),
            'null_mean': null_mean.astype(np.float32),
            'null_std':  null_std.astype(np.float32),
        }
    return out


# ── Public API ─────────────────────────────────────────────────────────

def compute_regional_wavelet_coherence(p1_bl, p2_bl, regions=None, fs=30.0,
                                        n_surrogates=200, smooth_s=0.5,
                                        seed=42, device='auto', freqs=None,
                                        pooling='au_pooled'):
    """Wavelet coherence per anatomical region with surrogate z.

    Two pooling strategies are supported (keyword `pooling`):

    * 'au_pooled' (default) — sums cross-/auto-spectra across all AUs in a
      region, then z-scores against circular-shift surrogates. Preserves
      omnibus AU information, but on regions with many AUs and signal in only
      a few (e.g. 27-AU mouth pool, smile signal on 2 AUs) the noise from the
      uninformative AUs dilutes the cross-spectrum and inflates the auto-
      spectra, lowering coherence.

    * 'pc1' — fits PC1 per region per participant on the raw AU panel, then
      computes univariate wavelet coherence on the two PC1 timeseries.
      Concentrates whatever co-varies in the region into a single mode, so
      smile-cluster co-activation (smile + dimple + jaw open) is recovered as
      one signal rather than 27 partially-correlated copies. Loadings are
      readable post-hoc via `compute_regional_pca` for interpretation.

    Args:
        p1_bl, p2_bl: (T, n_ch) blendshape arrays at fs (low-pass filtered).
        regions: dict region_name -> AU indices. Default: AU_REGIONS.
        fs: sampling rate (default 30 Hz).
        n_surrogates: number of circular-shift surrogates (default 200).
        smooth_s: temporal smoothing for coherence (seconds).
        seed: rng seed for surrogate shifts.
        device: 'cuda', 'cpu', or 'auto'.
        freqs: frequency axis for CWT (default 30 log-spaced 0.3-8 Hz).

    Returns:
        dict region_name -> {
            'z':             (n_freqs, T) z-scored coherence,
            'real_coh':      (n_freqs, T) raw coherence,
            'null_mean':     (n_freqs, T),
            'null_std':      (n_freqs, T),
            'band_z_mean':   {state, expression, speech} -> float (mean z in band),
            'band_z_max':    {state, expression, speech} -> float (max z in band),
            'band_z_ts':     {band -> (T,)} band-averaged z timecourse (mean over band freqs),
            'freqs':         (n_freqs,),
            'au_indices':    list of AU indices in region,
        }
    """
    if regions is None:
        regions = AU_REGIONS
    if pooling not in ('au_pooled', 'pc1'):
        raise ValueError(f"Unknown pooling: {pooling!r}")

    T, n_ch = p1_bl.shape
    n_ch2 = p2_bl.shape[1]
    n_ch = min(n_ch, n_ch2)
    p1_bl = p1_bl[:, :n_ch]
    p2_bl = p2_bl[:, :n_ch]

    # Filter region AU indices to valid columns
    region_au_lists = {}
    for name, aus in regions.items():
        valid = [a for a in aus if 0 <= a < n_ch]
        if len(valid) >= 1:
            region_au_lists[name] = valid

    if not region_au_lists:
        return {}

    # ---------------------------------------------------------------- pc1
    if pooling == 'pc1':
        return _regional_coherence_pc1(
            p1_bl, p2_bl, region_au_lists, fs=fs,
            n_surrogates=n_surrogates, smooth_s=smooth_s,
            seed=seed, device=device, freqs=freqs,
        )

    # ---------------------------------------------------------------- au_pooled
    # Compute CWT once per participant on the FULL panel (we slice by region
    # inside the multi-region pass).  Avoid redundant CWT.
    # The CWT operates on (T, n_ch) — passing the full panel keeps original AU
    # indexing intact, which simplifies per-region slicing downstream.
    scal_p1 = compute_au_cwt(p1_bl, fs=fs, freqs=freqs, device=device)
    scal_p2 = compute_au_cwt(p2_bl, fs=fs, freqs=freqs, device=device)
    freqs_axis = scal_p1.freqs

    rng = np.random.default_rng(seed)
    Tcwt = scal_p1.coeffs.shape[1]
    min_shift = max(1, int(0.1 * Tcwt))
    if Tcwt - min_shift <= min_shift:
        # Very short window — fallback shifts
        shifts = rng.integers(1, max(2, Tcwt - 1), size=n_surrogates)
    else:
        shifts = rng.integers(min_shift, Tcwt - min_shift, size=n_surrogates)

    sigma = smooth_s * fs
    use_gpu = (device == 'cuda' or (device == 'auto' and _HAS_TORCH))
    if use_gpu:
        per_region_raw = _multi_region_surrogate_z_gpu(
            scal_p1.coeffs, scal_p2.coeffs, region_au_lists, sigma, shifts)
    else:
        per_region_raw = _multi_region_surrogate_z_cpu(
            scal_p1.coeffs, scal_p2.coeffs, region_au_lists, sigma, shifts)

    bands = {'state': BAND_STATE, 'expression': BAND_EXPRESSION, 'speech': BAND_SPEECH}

    out = {}
    for region_name, raw in per_region_raw.items():
        z = (raw['real_coh'] - raw['null_mean']) / (raw['null_std'] + 1e-8)

        band_z_mean = {}
        band_z_max = {}
        band_z_ts_mean = {}
        band_z_ts_max = {}
        for bname, (lo, hi) in bands.items():
            mask = (freqs_axis >= lo) & (freqs_axis < hi)
            if mask.sum() == 0:
                band_z_mean[bname] = 0.0
                band_z_max[bname] = 0.0
                band_z_ts_mean[bname] = np.zeros(z.shape[1], dtype=np.float32)
                band_z_ts_max[bname] = np.zeros(z.shape[1], dtype=np.float32)
            else:
                band_z_mean[bname] = float(z[mask].mean())
                band_z_max[bname] = float(z[mask].max())
                # Per-timepoint band aggregations.  V11 uses max across
                # freqs in band for bl_expr; mean is gentler / more stable.
                band_z_ts_mean[bname] = z[mask].mean(axis=0).astype(np.float32)
                band_z_ts_max[bname] = z[mask].max(axis=0).astype(np.float32)

        out[region_name] = {
            'z': z.astype(np.float32),
            'real_coh': raw['real_coh'],
            'null_mean': raw['null_mean'],
            'null_std': raw['null_std'],
            'band_z_mean': band_z_mean,
            'band_z_max':  band_z_max,
            # Default 'band_z_ts' = max-across-freqs (matches V11 bl_expr).
            'band_z_ts':       band_z_ts_max,
            'band_z_ts_mean':  band_z_ts_mean,
            'band_z_ts_max':   band_z_ts_max,
            'freqs': freqs_axis,
            'au_indices': region_au_lists[region_name],
        }

    return out


# ── PC1-pooled regional coherence (alternative to AU-pooled) ──────────

def _regional_pc1_timeseries(bl_data, region_au_lists):
    """Per-region PC1 timeseries (one column per region) plus loadings.

    Returns:
        ts:       (T, n_regions) PC1 projections in the order of region_au_lists.
        loadings: dict region_name -> (k_aus,) loading vector.
        means:    dict region_name -> (k_aus,) per-AU means (centering offset).
        ordered_names: list of region names in the column order of ts.
    """
    cols = []
    loadings = {}
    means = {}
    names = []
    for region_name, aus in region_au_lists.items():
        panel = bl_data[:, aus].astype(np.float64)
        if panel.shape[1] < 1 or panel.std() < 1e-10:
            continue
        m = panel.mean(axis=0)
        X = panel - m[None, :]
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        # Sign-fix loading so the largest |loading| is positive — keeps
        # cross-participant signs comparable when computing coherence.
        v1 = Vt[0, :]
        if v1[np.argmax(np.abs(v1))] < 0:
            v1 = -v1
            U[:, 0] = -U[:, 0]
        ts1 = (U[:, 0] * S[0]).astype(np.float32)
        cols.append(ts1)
        loadings[region_name] = v1.astype(np.float32)
        means[region_name] = m.astype(np.float32)
        names.append(region_name)

    if not cols:
        return None, {}, {}, []
    ts = np.column_stack(cols).astype(np.float32)
    return ts, loadings, means, names


def _regional_coherence_pc1(p1_bl, p2_bl, region_au_lists, fs, n_surrogates,
                              smooth_s, seed, device, freqs):
    """Univariate wavelet coherence on per-region PC1 timeseries."""
    ts1, loadings1, means1, names = _regional_pc1_timeseries(p1_bl, region_au_lists)
    ts2, loadings2, means2, _     = _regional_pc1_timeseries(p2_bl, region_au_lists)
    if ts1 is None or ts2 is None or ts1.shape[1] == 0 or ts2.shape[1] == 0:
        return {}

    # Match column order across participants (same regions in same order)
    common = [r for r in names if r in loadings2]
    if not common:
        return {}
    idx1 = [names.index(r) for r in common]
    idx2_names = list(loadings2.keys())
    idx2 = [idx2_names.index(r) for r in common]
    ts1 = ts1[:, idx1]
    # Need ts2 in the same order
    ts2_ordered_cols = []
    # Recompute ts2 with explicit column order
    cols2 = []
    for r in common:
        panel = p2_bl[:, region_au_lists[r]].astype(np.float64)
        m = panel.mean(axis=0)
        X = panel - m[None, :]
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            cols2.append(np.zeros(p2_bl.shape[0], dtype=np.float32))
            continue
        v1 = Vt[0, :]
        if v1[np.argmax(np.abs(v1))] < 0:
            v1 = -v1
            U[:, 0] = -U[:, 0]
        cols2.append((U[:, 0] * S[0]).astype(np.float32))
    ts2 = np.column_stack(cols2).astype(np.float32)

    # CWT on the (T, n_regions) panels — one channel per region.
    scal_p1 = compute_au_cwt(ts1, fs=fs, freqs=freqs, device=device)
    scal_p2 = compute_au_cwt(ts2, fs=fs, freqs=freqs, device=device)
    freqs_axis = scal_p1.freqs

    out = {}
    for col_i, region_name in enumerate(common):
        sgz = surrogate_coherence_z(
            scal_p1, scal_p2,
            n_surrogates=n_surrogates, smooth_s=smooth_s,
            aus=[col_i], seed=seed + col_i,
            device=device,
        )
        z = sgz['z']
        out[region_name] = _build_region_summary(
            z, sgz['real_coh'], sgz['null_mean'], sgz['null_std'],
            freqs_axis, region_au_lists[region_name],
            extra={
                'pc1_loading': loadings1.get(region_name),
                'pc1_loading_p2': loadings2.get(region_name),
            },
        )
    return out


def _build_region_summary(z, real_coh, null_mean, null_std, freqs_axis,
                            au_indices, extra=None):
    bands = {'state': BAND_STATE, 'expression': BAND_EXPRESSION, 'speech': BAND_SPEECH}
    band_z_mean = {}
    band_z_max = {}
    band_z_ts_mean = {}
    band_z_ts_max = {}
    for bname, (lo, hi) in bands.items():
        mask = (freqs_axis >= lo) & (freqs_axis < hi)
        if mask.sum() == 0:
            band_z_mean[bname] = 0.0
            band_z_max[bname] = 0.0
            band_z_ts_mean[bname] = np.zeros(z.shape[1], dtype=np.float32)
            band_z_ts_max[bname] = np.zeros(z.shape[1], dtype=np.float32)
        else:
            band_z_mean[bname] = float(z[mask].mean())
            band_z_max[bname] = float(z[mask].max())
            band_z_ts_mean[bname] = z[mask].mean(axis=0).astype(np.float32)
            band_z_ts_max[bname] = z[mask].max(axis=0).astype(np.float32)
    summary = {
        'z': z.astype(np.float32),
        'real_coh': real_coh.astype(np.float32),
        'null_mean': null_mean.astype(np.float32),
        'null_std': null_std.astype(np.float32),
        'band_z_mean': band_z_mean,
        'band_z_max': band_z_max,
        'band_z_ts': band_z_ts_max,
        'band_z_ts_mean': band_z_ts_mean,
        'band_z_ts_max': band_z_ts_max,
        'freqs': freqs_axis,
        'au_indices': au_indices,
    }
    if extra:
        summary.update(extra)
    return summary


# ── Per-region BL injection (replaces V82's AFFECT_AUS-locked variant) ──

def inject_bl_regional(p1_bl, p2_bl, kappa, au_subset, lag_s, gate,
                        seed=42, fs=30.0, expression_band=(0.5, 2.0),
                        boost=3.0, inject_activity=True,
                        secondary_aus=None, secondary_boost=1.5):
    """Inject expression-band coupling on an AU subset (and optional secondaries).

    Mirrors inject_bl_v82's continuous expression-band signal-mixing strategy
    but is NOT restricted to AFFECT_AUS — works on any AU indices including
    brow, eye, cheek, nose. Continuous mixing creates sustained wavelet
    coherence detectable by CWT + surrogate z.

    Two-tier injection (matches inject_bl_v82's pattern):
      * `au_subset` AUs receive `boost` (default 3.0). These are the "primary"
        AUs of the simulated facial action.
      * `secondary_aus` AUs receive `secondary_boost` (default 1.5). These
        co-activate weakly along with the primary AUs, simulating natural
        facial-action coupling (e.g. smile primaries co-activate dimples and
        upper-lip raisers). This makes the injected signal regionally
        distributed rather than concentrated on 2-4 AUs, which is what the
        regional coherence detector expects to see.

    P1 is unchanged; P2 receives a lagged, gate-modulated expression-band
    copy of P1's panel on the affected AUs.

    Args:
        p1_bl, p2_bl: (T, C) raw blendshapes at fs.
        kappa: coupling strength.
        au_subset: primary AU indices (full boost).
        lag_s: coupling lag (seconds).
        gate: (T,) coupling gate in [0, 1].
        seed: rng seed (reserved for future use).
        fs: sampling rate.
        expression_band: (lo, hi) Hz for narrowband filter.
        boost: primary-AU coupling amplification (default 3.0).
        secondary_aus: list of AUs receiving the weaker (secondary) coupling.
            Pass None or [] to disable. Indices in au_subset will be skipped
            from secondaries automatically.
        secondary_boost: amplification for secondary AUs (default 1.5).
        inject_activity: if True and an activity channel (col C-1, C >= 53)
            exists, also inject a coupled signal there for bl_activity_conc.

    Returns:
        p1_out, p2_out: (T, C) float32 with injection applied to p2.
    """
    from scipy.signal import butter, sosfiltfilt

    T, C = p1_bl.shape
    p1_out = p1_bl.astype(np.float32, copy=True)
    p2_out = p2_bl.astype(np.float64, copy=True)

    lag_samp = max(0, int(lag_s * fs))
    lo, hi = expression_band
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq * 0.95
    sos = butter(4, [lo / nyq, hi / nyq], btype='band', output='sos')

    g = np.asarray(gate, dtype=np.float64)
    if len(g) < T:
        g = np.concatenate([g, np.zeros(T - len(g))])
    g = g[:T]

    primary_set = set(au_subset)
    secondary_set = (set(secondary_aus) - primary_set) if secondary_aus else set()

    def _inject_one(au, b):
        if au < 0 or au >= C:
            return
        p1_expr = sosfiltfilt(sos, p1_bl[:, au].astype(np.float64))
        p1_lagged = np.roll(p1_expr, lag_samp)
        if lag_samp > 0:
            p1_lagged[:lag_samp] = 0.0
        p1_std = max(p1_expr.std(), 1e-6)
        p2_std = max(p2_bl[:, au].astype(np.float64).std(), 1e-6)
        alpha = np.clip(kappa * b * g, 0, 0.95)
        p2_out[:, au] += alpha * p1_lagged * (p2_std / p1_std)

    for au in primary_set:
        _inject_one(au, boost)
    for au in secondary_set:
        _inject_one(au, secondary_boost)

    # Activity channel injection (mirrors inject_bl_v82) — only when present.
    act_ch = C - 1
    if inject_activity and act_ch >= 52:
        p1_act = p1_bl[:, act_ch].astype(np.float64)
        p2_act = p2_bl[:, act_ch].astype(np.float64)
        p1_act_filt = sosfiltfilt(sos, p1_act)
        p1_act_lagged = np.roll(p1_act_filt, lag_samp)
        if lag_samp > 0:
            p1_act_lagged[:lag_samp] = 0.0
        act_alpha = np.clip(kappa * 2.0 * g, 0, 0.95)
        p1_a_std = max(p1_act_filt.std(), 1e-6)
        p2_a_std = max(p2_act.std(), 1e-6)
        p2_out[:, act_ch] += act_alpha * p1_act_lagged * (p2_a_std / p1_a_std)

    return p1_out, p2_out.astype(np.float32)


# ── Per-region scenario registry (battery configuration) ───────────────

#
# Realistic region-wide co-activation patterns.
# au_subset = primary action AUs (full coupling boost = 3.0 in injection).
# secondary_aus = co-activated AUs in same anatomical region (boost = 1.5).
# Together these distribute the injected signal across most of the region,
# matching how real facial actions activate multiple synergistic AUs (smile
# = corner-up + dimple + upper-lip + cheek squint, etc.). Without secondary
# co-activation the regional coherence detector dilutes a 2-AU signal across
# 27 mouth AUs and detection collapses (verified in smoke testing).
REGIONAL_SCENARIOS = {
    'R_mouth_smile': {
        'target_region': 'mouth',
        # Primary: mouthSmileL/R + mouthDimpleL/R
        'au_subset':     [28, 29, 44, 45],
        # Secondary: jawOpen + upperUp L/R + lower-down L/R (laughter coactives)
        'secondary_aus': [25, 34, 35, 48, 49],
        'lag_s': 0.5,
        'rationale': 'realistic smile co-activation across mouth region',
    },
    'R_mouth_frown': {
        'target_region': 'mouth',
        # Primary: mouthFrownL/R + mouthShrugLower/Upper
        'au_subset':     [30, 31, 42, 43],
        # Secondary: pressL/R + rollLower/Upper (sad mouth co-activations)
        'secondary_aus': [36, 37, 40, 41],
        'lag_s': 4.0,
        'rationale': 'realistic frown co-activation across mouth region',
    },
    'R_brow_furrow': {
        'target_region': 'brow',
        # Primary: browDownL/R + browInnerUp (concentration / concern)
        'au_subset':     [1, 2, 3],
        # Secondary: browOuterUpL/R (empathic / surprised co-activation)
        'secondary_aus': [4, 5],
        'lag_s': 2.0,
        'rationale': 'brow-wide concentration / concern',
    },
    'R_eye_squint': {
        'target_region': 'eye',
        # Primary: eyeSquintL/R + eyeBlinkL/R (Duchenne marker)
        'au_subset':     [9, 10, 19, 20],
        # Secondary: eyeWideL/R (negative co-activation, but still expression-band)
        'secondary_aus': [21, 22],
        'lag_s': 1.0,
        'rationale': 'eye-wide squint + blink co-activation (Duchenne marker)',
    },
    'R_cheek_squint': {
        'target_region': 'cheek',
        # All cheek AUs (only 3 total — region is naturally small)
        'au_subset':     [7, 8],            # cheekSquintL/R primary
        'secondary_aus': [6],               # cheekPuff secondary
        'lag_s': 0.5,
        'rationale': 'cheek-wide squint + puff (smile-coupled)',
    },
    'R_nose_sneer': {
        'target_region': 'nose',
        # Only 2 AUs in region, both primary
        'au_subset':     [50, 51],
        'secondary_aus': [],
        'lag_s': 1.0,
        'rationale': 'nose-wide sneer',
    },
    'R_duchenne': {
        'target_region': 'mouth+cheek',
        # Cross-region positive control: smile mouth AUs + cheek squint
        'au_subset':     [28, 29, 44, 45, 7, 8],
        'secondary_aus': [6, 25, 48, 49],
        # Lag = 0.5s: real Duchenne mouth and cheek AUs co-activate within
        # ~100-200ms (Ekman 1990).  V82's 2.7s lag value was for inter-
        # participant mimicry response time, which is a different timescale
        # and exceeds the bandwidth of expression-band wavelet coherence
        # (autocorrelation of a 0.5-2 Hz signal at 2.7s lag is severely
        # reduced).  Verified empirically: 2.7s lag → AUC=0.50; 0.5s → strong.
        'lag_s': 0.5,
        'rationale': 'cross-region: mouth AND cheek co-activate (Duchenne)',
    },
}
