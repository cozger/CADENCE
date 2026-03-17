"""Surrogate generation: circular time-shift, Fourier phase randomization (GPU-native), IBI/block shuffle."""

import math
import numpy as np
import torch


def circular_shift_surrogate_batched(data, n_surrogates, min_shift_frac=0.1,
                                     generator=None):
    """Generate K circular time-shift surrogates (GPU-native).

    Circular shift preserves ALL statistical properties of the input
    (amplitude distribution, autocorrelation, cross-channel correlations)
    while destroying temporal alignment with other signals.

    Parameters:
        data: (1, C, N) tensor on any device
        n_surrogates: number of surrogates K to generate
        min_shift_frac: minimum shift as fraction of N (default 0.1 = 10%)
        generator: optional torch.Generator for reproducibility

    Returns:
        surrogates: (K, C, N) tensor -- circularly shifted versions
    """
    N = data.shape[-1]
    min_shift = max(1, int(min_shift_frac * N))
    max_shift = N - min_shift

    if min_shift >= max_shift:
        min_shift = 1
        max_shift = N - 1

    if generator is not None:
        shifts = torch.randint(min_shift, max_shift + 1, (n_surrogates,),
                               generator=generator, device=data.device)
    else:
        shifts = torch.randint(min_shift, max_shift + 1, (n_surrogates,),
                               device=data.device)

    # Vectorized gather: single kernel replaces K sequential torch.roll calls.
    # Index tensor is int32 (not int64) to halve memory: K×N×4 bytes.
    data_sq = data.squeeze(0)  # (C, N)
    base_idx = torch.arange(N, device=data.device, dtype=torch.int32)
    # (K, N) index matrix: each row is arange shifted by -shift[k] mod N
    indices = (base_idx.unsqueeze(0) - shifts.to(torch.int32).unsqueeze(1)) % N
    # Gather along time dim: data_sq[:, indices] → (C, K, N), permute to (K, C, N)
    surrogates = data_sq[:, indices.long()].permute(1, 0, 2)
    return surrogates


def fourier_surrogate(data, seed=None):
    """
    Generate a surrogate time series preserving power spectrum
    but destroying temporal structure (phase relationships).

    Parameters:
        data: (N, C) numpy array

    Returns:
        surrogate: (N, C) phase-randomized version
    """
    rng = np.random.RandomState(seed)
    surrogate = np.empty_like(data)

    for ch in range(data.shape[1]):
        spectrum = np.fft.rfft(data[:, ch])
        random_phases = np.exp(1j * rng.uniform(0, 2 * np.pi, len(spectrum)))
        random_phases[0] = 1.0
        if len(data) % 2 == 0:
            random_phases[-1] = 1.0
        surrogate[:, ch] = np.fft.irfft(spectrum * random_phases, n=len(data))

    return surrogate


def fourier_surrogate_gpu(tensor, seed=None):
    """
    GPU-native Fourier phase randomization using torch.fft.

    Parameters:
        tensor: (B, n_channels, n_samples) on any device

    Returns:
        surrogate: same shape and device, phase-randomized
    """
    n_samples = tensor.shape[-1]
    spectrum = torch.fft.rfft(tensor, dim=-1)

    if seed is not None:
        gen = torch.Generator(device=tensor.device)
        gen.manual_seed(seed)
        random_angles = torch.rand(spectrum.shape, device=tensor.device,
                                   dtype=tensor.dtype, generator=gen) * (2 * math.pi)
    else:
        random_angles = torch.rand(spectrum.shape, device=tensor.device,
                                   dtype=tensor.dtype) * (2 * math.pi)

    random_phases = torch.polar(torch.ones_like(random_angles), random_angles)

    random_phases[..., 0] = 1.0
    if n_samples % 2 == 0:
        random_phases[..., -1] = 1.0

    return torch.fft.irfft(spectrum * random_phases, n=n_samples)


def fourier_surrogate_gpu_batched(tensor, n_surrogates, base_seed=0,
                                   sync_channels=False):
    """
    Generate K Fourier surrogates in a single batched FFT/iFFT.

    Input: (1, C, N) tensor -> Output: (K, C, N) tensor

    Parameters:
        tensor: (1, C, N) on any device
        n_surrogates: number of surrogates K
        base_seed: base seed; surrogate i uses seed = base_seed + i
        sync_channels: if True, same random phases across all channels

    Returns:
        surrogates: (K, C, N) phase-randomized versions
    """
    C, N = tensor.shape[1], tensor.shape[2]
    device = tensor.device
    dtype = tensor.dtype

    spectrum = torch.fft.rfft(tensor, dim=-1)  # (1, C, F)
    F = spectrum.shape[-1]

    if sync_channels:
        all_angles = torch.empty(n_surrogates, 1, F, device=device, dtype=dtype)
        for i in range(n_surrogates):
            gen = torch.Generator(device=device)
            gen.manual_seed(base_seed + i)
            all_angles[i] = torch.rand(1, F, device=device, dtype=dtype, generator=gen) * (2 * math.pi)
        all_angles = all_angles.expand(n_surrogates, C, F).contiguous()
    else:
        all_angles = torch.empty(n_surrogates, C, F, device=device, dtype=dtype)
        for i in range(n_surrogates):
            gen = torch.Generator(device=device)
            gen.manual_seed(base_seed + i)
            all_angles[i] = torch.rand(C, F, device=device, dtype=dtype, generator=gen) * (2 * math.pi)

    random_phases = torch.polar(torch.ones_like(all_angles), all_angles)

    random_phases[:, :, 0] = 1.0
    if N % 2 == 0:
        random_phases[:, :, -1] = 1.0

    shifted = spectrum * random_phases
    return torch.fft.irfft(shifted, n=N)


def fourier_surrogate_tensors(data_dict, seed=None):
    """
    Apply Fourier phase randomization to a dict of tensors.

    Parameters:
        data_dict: dict of {modality: (B, n_channels, n_samples)} tensors

    Returns:
        surrogate_dict: same structure with phase-randomized data
    """
    surrogate_dict = {}

    for i, (mod, tensor) in enumerate(data_dict.items()):
        mod_seed = (seed * 1000 + i) if seed is not None else None

        if tensor.is_cuda:
            surrogate_dict[mod] = fourier_surrogate_gpu(tensor, seed=mod_seed)
        else:
            arr = tensor.numpy()
            B = arr.shape[0]
            surr_arr = np.empty_like(arr)
            for b in range(B):
                s = (mod_seed * 100 + b) if mod_seed is not None else None
                surr_arr[b] = fourier_surrogate(arr[b].T, seed=s).T
            surrogate_dict[mod] = torch.tensor(surr_arr, dtype=tensor.dtype)
            del arr, surr_arr

    return surrogate_dict


def block_shuffle_surrogate(tensor, block_duration_range=(2.0, 5.0),
                            sample_rate=256, seed=None):
    """
    Block-shuffle surrogate: shuffle contiguous blocks to destroy phase
    alignment while preserving within-block temporal structure.

    Parameters:
        tensor: (B, n_channels, n_samples) tensor
        block_duration_range: (min_sec, max_sec) block durations
        sample_rate: samples per second
        seed: random seed

    Returns:
        surrogate: same shape, block-shuffled
    """
    B, C, T = tensor.shape
    rng = np.random.RandomState(seed)
    surrogate = tensor.clone()

    min_samples = max(1, int(block_duration_range[0] * sample_rate))
    max_samples = max(min_samples + 1, int(block_duration_range[1] * sample_rate))

    for b in range(B):
        blocks = []
        pos = 0
        while pos < T:
            blen = rng.randint(min_samples, max_samples)
            blen = min(blen, T - pos)
            blocks.append((pos, pos + blen))
            pos += blen

        order = list(range(len(blocks)))
        rng.shuffle(order)

        new_data = torch.empty_like(surrogate[b])
        write_pos = 0
        for idx in order:
            start, end = blocks[idx]
            blen = end - start
            new_data[:, write_pos:write_pos + blen] = surrogate[b, :, start:end]
            write_pos += blen
        surrogate[b] = new_data

    return surrogate


def ibi_shuffle_surrogate(ecg_features, block_size_range=(5, 10), seed=None):
    """
    Block-shuffle IBI-derived ECG features to destroy temporal coupling
    while preserving the HR distribution.

    Parameters:
        ecg_features: (B, n_channels, n_samples) ECG feature tensor
        block_size_range: (min_blocks, max_blocks) in output frames
        seed: random seed

    Returns:
        surrogate: same shape, block-shuffled
    """
    B, C, T = ecg_features.shape
    rng = np.random.RandomState(seed)
    surrogate = ecg_features.clone()

    min_block, max_block = block_size_range
    for b in range(B):
        blocks = []
        pos = 0
        while pos < T:
            blen = rng.randint(min_block, max_block + 1)
            blen = min(blen, T - pos)
            blocks.append((pos, pos + blen))
            pos += blen

        order = list(range(len(blocks)))
        rng.shuffle(order)

        new_data = torch.empty_like(surrogate[b])
        write_pos = 0
        for idx in order:
            start, end = blocks[idx]
            blen = end - start
            new_data[:, write_pos:write_pos + blen] = surrogate[b, :, start:end]
            write_pos += blen
        surrogate[b] = new_data

    return surrogate
