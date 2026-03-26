"""Quick import + config validation for V3 coherence TL."""
from cadence.significance.coherence_localization import (
    sliding_coherence, modality_coherence,
    aggregate_coherence_whitened, coherence_coupling_mask,
    coherence_surrogates_and_real, coherence_temporal_localization,
    masked_feature_breakdown,
)
print('coherence_localization: OK')

from cadence.data.eeg_coherence import (
    eeg_band_coherence, eeg_coherence_features, eeg_coherence_surrogates,
)
print('eeg_coherence: OK')

from cadence.significance.temporal_localization import (
    zscore_stouffer, zscore_stouffer_whitened,
    temporal_localization_pipeline,
)
print('temporal_localization: OK')

from cadence.config import load_config
cfg = load_config('configs/default.yaml')
print(f'temporal_localization.method: {cfg["temporal_localization"]["method"]}')
print(f'significance.skip_screening: {cfg["significance"]["skip_screening"]}')
print(f'slds.enabled: {cfg["slds"]["enabled"]}')
print(f'significance.timepoint.enabled: {cfg["significance"]["timepoint"]["enabled"]}')
print(f'eeg_coherence.enabled: {cfg.get("eeg_coherence", {}).get("enabled", False)}')

# Quick numerical test: sliding coherence on synthetic coupled signals
import numpy as np
rng = np.random.RandomState(42)
fs = 30.0
T = int(300 * fs)
t = np.arange(T) / fs

# Two signals: x2 = 0.5*x1(shifted) + noise during coupling windows
x1 = rng.randn(T)
x2 = rng.randn(T)
# Coupling in [50-100s] and [200-250s]
for s, e in [(50, 100), (200, 250)]:
    si, ei = int(s * fs), int(e * fs)
    lag = int(0.5 * fs)  # 0.5s lag
    x2[si+lag:ei+lag] += 0.5 * x1[si:ei]

coh, wt = sliding_coherence(x1, x2, fs, window_s=5.0, stride_s=0.5)
print(f'\nSliding coherence: {len(coh)} windows, '
      f'mean={coh.mean():.4f}, max={coh.max():.4f}')

# Check coherence is higher during coupling windows
coupled_mask = ((wt >= 50) & (wt <= 100)) | ((wt >= 200) & (wt <= 250))
coh_coupled = coh[coupled_mask].mean()
coh_null = coh[~coupled_mask].mean()
print(f'Coupled windows: mean_coh={coh_coupled:.4f}')
print(f'Null windows:    mean_coh={coh_null:.4f}')
print(f'Contrast:        {coh_coupled - coh_null:.4f}')
assert coh_coupled > coh_null, 'Coherence should be higher during coupling!'
print('\nAll V3 imports and basic test PASSED')
