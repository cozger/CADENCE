import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurtosis, entropy as _entropy


def dfa_exponent(x, min_win=4, max_win_frac=0.25):
    n = len(x)
    max_win = int(n * max_win_frac)
    if n < 16 or max_win < min_win:
        return None

    y = np.cumsum(x - x.mean())

    wins = np.unique(np.logspace(
        np.log10(min_win), np.log10(max_win), 20).astype(int))
    wins = wins[wins >= min_win]
    if len(wins) < 4:
        return None

    log_n = []
    log_f = []
    for w in wins:
        n_segs = n // w
        if n_segs < 1:
            continue
        rms_vals = np.empty(n_segs)
        t = np.arange(w, dtype=np.float64)
        for i in range(n_segs):
            seg = y[i * w:(i + 1) * w]
            coeffs = np.polyfit(t, seg, 1)
            residual = seg - (coeffs[0] * t + coeffs[1])
            rms_vals[i] = np.sqrt(np.mean(residual ** 2))
        f_n = rms_vals.mean()
        if f_n > 0:
            log_n.append(np.log(w))
            log_f.append(np.log(f_n))

    if len(log_n) < 4:
        return None

    alpha = np.polyfit(log_n, log_f, 1)[0]
    return round(float(alpha), 4)


def distributional_stats(x, min_n=10):
    null = {k: None for k in ('sd', 'skewness', 'kurtosis', 'entropy', 'dfa_exponent')}
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < min_n:
        return null

    sd = float(np.std(x))
    sk = float(_skew(x))
    ku = float(_kurtosis(x))

    n_bins = max(5, int(np.sqrt(len(x))))
    counts, _ = np.histogram(x, bins=n_bins)
    ent = float(_entropy(counts)) if counts.sum() > 0 else 0.0

    dfa = dfa_exponent(x)

    return {
        'sd': round(sd, 4),
        'skewness': round(sk, 4),
        'kurtosis': round(ku, 4),
        'entropy': round(ent, 4),
        'dfa_exponent': dfa,
    }
