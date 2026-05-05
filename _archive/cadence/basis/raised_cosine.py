"""Raised cosine basis functions for distributed lag regression.

Log-spaced centers provide dense coverage at short lags (where coupling
is strongest) and broader coverage at longer lags.
"""

import numpy as np


def raised_cosine_basis(n_basis, max_lag_s, min_lag_s=0.0, sample_rate=2.0,
                        log_spacing=True):
    """Generate a set of raised cosine basis functions over a lag axis.

    Each basis function: phi_j(s) = 0.5 * (1 + cos(pi * clamp((s - c_j)/w_j, -1, 1)))
    where c_j is the center and w_j is the half-width.

    Args:
        n_basis: Number of basis functions.
        max_lag_s: Maximum lag in seconds.
        min_lag_s: Minimum lag in seconds (default 0).
        sample_rate: Output sampling rate in Hz.
        log_spacing: If True, log-space centers (denser at short lags).
                     If False, linearly space centers.

    Returns:
        basis: (n_lag_samples, n_basis) array of basis function values.
        lag_times: (n_lag_samples,) array of lag times in seconds.
    """
    n_lag_samples = max(1, int((max_lag_s - min_lag_s) * sample_rate))
    lag_times = np.linspace(min_lag_s, max_lag_s, n_lag_samples)

    if n_basis < 1:
        return np.zeros((n_lag_samples, 0)), lag_times

    # Compute centers
    if log_spacing and max_lag_s > 0:
        # Log-space in [log(min+offset), log(max+offset)] then shift back
        offset = max(0.1, min_lag_s + 0.1)  # avoid log(0)
        log_min = np.log(min_lag_s + offset)
        log_max = np.log(max_lag_s + offset)
        centers = np.exp(np.linspace(log_min, log_max, n_basis)) - offset
    else:
        centers = np.linspace(min_lag_s, max_lag_s, n_basis)

    # Compute half-widths: each basis overlaps with neighbors
    if n_basis == 1:
        widths = np.array([(max_lag_s - min_lag_s) / 2 + 0.5])
    else:
        # Width = distance to nearest neighbor (ensures overlap)
        diffs = np.diff(centers)
        widths = np.empty(n_basis)
        widths[0] = diffs[0]
        widths[-1] = diffs[-1]
        widths[1:-1] = (diffs[:-1] + diffs[1:]) / 2
        # Ensure minimum width for numerical stability
        widths = np.maximum(widths, 0.5 / sample_rate)

    # Build basis matrix
    basis = np.zeros((n_lag_samples, n_basis))
    for j in range(n_basis):
        # Normalized distance from center
        z = (lag_times - centers[j]) / widths[j]
        # Raised cosine: nonzero only within [-1, 1]
        mask = np.abs(z) <= 1.0
        basis[mask, j] = 0.5 * (1.0 + np.cos(np.pi * z[mask]))

    return basis, lag_times


def multi_band_basis(bands, sample_rate=2.0):
    """Build a multi-band basis by concatenating separate raised cosine bands.

    Each band covers a distinct lag range. Bands can be non-contiguous (gaps
    are fine — the gap simply has no basis coverage, meaning no coupling at
    those lags is modelled).

    Args:
        bands: List of dicts, each with keys:
            n_basis: int, number of basis functions in this band.
            min_lag_seconds: float, start of lag range.
            max_lag_seconds: float, end of lag range.
            log_spacing: bool (default True).
        sample_rate: Shared sample rate in Hz.

    Returns:
        basis: (n_lag_samples, total_n_basis) array.
               n_lag_samples covers the full range from min(all min_lags)
               to max(all max_lags).
        lag_times: (n_lag_samples,) array of lag times in seconds.
        band_slices: List of (start_col, end_col) tuples identifying which
                     columns belong to each band.
    """
    if not bands:
        raise ValueError("At least one band required")

    global_min = min(b['min_lag_seconds'] for b in bands)
    global_max = max(b['max_lag_seconds'] for b in bands)
    n_lag_samples = max(1, int((global_max - global_min) * sample_rate))
    lag_times = np.linspace(global_min, global_max, n_lag_samples)

    columns = []
    band_slices = []
    col_offset = 0

    for band in bands:
        # Build this band's basis on its own lag range
        b, b_lags = raised_cosine_basis(
            n_basis=band['n_basis'],
            max_lag_s=band['max_lag_seconds'],
            min_lag_s=band['min_lag_seconds'],
            sample_rate=sample_rate,
            log_spacing=band.get('log_spacing', True),
        )

        # Embed into the global lag grid
        global_basis = np.zeros((n_lag_samples, band['n_basis']))
        for i, t in enumerate(lag_times):
            # Find nearest lag in this band's grid
            if t < b_lags[0] or t > b_lags[-1]:
                continue
            idx = np.searchsorted(b_lags, t)
            idx = min(idx, len(b_lags) - 1)
            # Linear interpolation between nearest samples
            if idx > 0 and idx < len(b_lags):
                t0, t1 = b_lags[idx - 1], b_lags[idx]
                if t1 > t0:
                    frac = (t - t0) / (t1 - t0)
                    global_basis[i] = (1 - frac) * b[idx - 1] + frac * b[idx]
                else:
                    global_basis[i] = b[idx]
            else:
                global_basis[i] = b[idx]

        columns.append(global_basis)
        band_slices.append((col_offset, col_offset + band['n_basis']))
        col_offset += band['n_basis']

    basis = np.hstack(columns)
    return basis, lag_times, band_slices


def basis_summary(basis, lag_times):
    """Print summary of basis function properties.

    Args:
        basis: (n_lag_samples, n_basis) basis matrix.
        lag_times: (n_lag_samples,) lag times in seconds.
    """
    n_samples, n_basis = basis.shape
    print(f"Basis: {n_basis} functions over {lag_times[0]:.2f}-{lag_times[-1]:.2f}s "
          f"({n_samples} samples)")

    # Check smoothness: max abs second derivative
    for j in range(n_basis):
        peak_idx = np.argmax(basis[:, j])
        peak_lag = lag_times[peak_idx]
        support = np.sum(basis[:, j] > 0.01)
        print(f"  phi_{j}: peak at {peak_lag:.2f}s, support={support} samples")

    # Sum check
    col_sum = basis.sum(axis=1)
    print(f"  Sum range: [{col_sum.min():.3f}, {col_sum.max():.3f}] "
          f"(ideal ~1.0 in overlap region)")
