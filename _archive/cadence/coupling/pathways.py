"""Pathway definitions for modality-level and feature-level coupling analysis.

Phase 1: 32 modality-level pathways (4 source x 4 target x 2 directions).
  For a given direction (A->B), we have 4x4 = 16 pathways.
  But we only need source_mod -> target_mod (not self), so 4*4 = 16 total
  including self-prediction pathways which serve as controls.

Phase 2: Feature-level decomposition within significant pathways.

V2 pipeline adds wavelet EEG, inter-brain, and expanded blendshape/cardiac modalities.
"""

from cadence.constants import (
    MODALITY_ORDER, MODALITY_SPECS, MODALITY_BASE_CH,
    ECG_FEATURE_NAMES, POSE_SEGMENT_MAP, BLENDSHAPE_SEGMENT_MAP,
    EEG_FEATURE_NAMES_V6,
    MODALITY_ORDER_V2, MODALITY_SPECS_V2, INTERBRAIN_MODALITY,
    WAVELET_FEATURE_NAMES, INTERBRAIN_FEATURE_NAMES,
    BL_FEATURE_NAMES_V2, ECG_FEATURE_NAMES_V2,
)


def get_modality_pathways():
    """Get all 16 modality-level pathways for one direction.

    Returns:
        list of (source_mod, target_mod) tuples.
        Each is a pathway to test: does source modality predict target?

    Note: includes cross-modal AND self-prediction pathways.
    Self pathways (eeg->eeg) measure autoregressive predictability.
    Cross pathways (eeg->ecg) measure cross-modal coupling.
    """
    pathways = []
    for src in MODALITY_ORDER:
        for tgt in MODALITY_ORDER:
            pathways.append((src, tgt))
    return pathways


def get_cross_modal_pathways():
    """Get only cross-modal pathways (source != target), 12 total.

    These are the interesting pathways for coupling analysis.
    Self pathways are used only as AR baselines.
    """
    return [(s, t) for s, t in get_modality_pathways() if s != t]


def get_feature_groups(modality):
    """Get feature-level decomposition groups for a modality.

    Args:
        modality: One of MODALITY_ORDER.

    Returns:
        dict mapping group_name -> (start_ch, end_ch) or group_name -> [ch_list]
    """
    if modality == 'eeg_features':
        # Each EEG feature is its own group
        groups = {}
        for i, name in enumerate(EEG_FEATURE_NAMES_V6):
            groups[name] = (i, i + 1)
        return groups
    elif modality == 'ecg_features':
        groups = {}
        for i, name in enumerate(ECG_FEATURE_NAMES):
            groups[name] = (i, i + 1)
        return groups
    elif modality == 'pose_features':
        return {name: rng for name, rng in POSE_SEGMENT_MAP.items()}
    elif modality == 'blendshapes':
        return {name: channels for name, channels in BLENDSHAPE_SEGMENT_MAP.items()}
    else:
        # Unknown modality — treat whole thing as one group
        n_ch = MODALITY_SPECS.get(modality, (1, 1))[0]
        return {modality: (0, n_ch)}


def get_pathway_n_predictors(source_mod, n_basis, ar_order, target_mod):
    """Compute number of predictors for a given pathway.

    Full model: n_basis * n_source_channels + ar_order * n_target_channels
    Restricted model: ar_order * n_target_channels

    Args:
        source_mod: Source modality name.
        n_basis: Number of basis functions.
        ar_order: AR lag order.
        target_mod: Target modality name.

    Returns:
        (p_full, p_restricted) tuple.
    """
    specs = MODALITY_SPECS_V2 if source_mod in MODALITY_SPECS_V2 else MODALITY_SPECS
    tgt_specs = MODALITY_SPECS_V2 if target_mod in MODALITY_SPECS_V2 else MODALITY_SPECS
    n_src = specs.get(source_mod, MODALITY_SPECS.get(source_mod, (1, 1)))[0]
    n_tgt = tgt_specs.get(target_mod, MODALITY_SPECS.get(target_mod, (1, 1)))[0]

    p_restricted = ar_order * n_tgt
    p_full = n_basis * n_src + p_restricted

    return p_full, p_restricted


# ---------------------------------------------------------------------------
# V2 pathway definitions
# ---------------------------------------------------------------------------

def get_modality_pathways_v2():
    """Get all v2 modality-level pathways for one direction.

    Includes participant-owned modalities (eeg_wavelet, blendshapes_v2,
    pose_features) as sources and all four modalities as targets, plus
    eeg_interbrain as source-only.

    ECG is excluded as a source modality — autonomic state is modeled
    as an RMSSD moderator that modulates coupling strength. ECG remains
    as a target (other modalities can still predict ECG changes).

    Returns:
        list of (source_mod, target_mod) tuples.
    """
    # Source modalities: exclude ecg_features_v2
    source_mods = [m for m in MODALITY_ORDER_V2 if m != 'ecg_features_v2']

    pathways = []
    # Source modalities (excluding ECG) -> all targets (3x4 = 12)
    for src in source_mods:
        for tgt in MODALITY_ORDER_V2:
            pathways.append((src, tgt))
    # Inter-brain as source -> all participant targets
    for tgt in MODALITY_ORDER_V2:
        pathways.append((INTERBRAIN_MODALITY, tgt))
    return pathways


def get_cross_modal_pathways_v2():
    """Get only cross-modal v2 pathways (source != target)."""
    return [(s, t) for s, t in get_modality_pathways_v2() if s != t]


def get_pathway_category(src_mod, tgt_mod, config=None):
    """Determine pathway speed category: 'fast', 'medium', or 'slow'.

    Cardiac targets are always 'slow'. Other assignments come from config
    or use 'medium' as default.

    Args:
        src_mod: source modality name
        tgt_mod: target modality name
        config: optional config dict with pathway_category mapping

    Returns:
        str: 'fast', 'medium', or 'slow'
    """
    # Cardiac targets are always slow
    if tgt_mod in ('ecg_features', 'ecg_features_v2'):
        return 'slow'

    # Check explicit config mapping
    if config is not None:
        cat_cfg = config.get('pathway_category', {})
        key = f'{src_mod}->{tgt_mod}'
        if key in cat_cfg:
            return cat_cfg[key]
        # Check wildcard patterns
        for pattern, cat in cat_cfg.items():
            if pattern == 'default':
                continue
            if pattern.startswith('*->') and pattern[3:] == tgt_mod:
                return cat
            if pattern.endswith('->*') and pattern[:-3] == src_mod:
                return cat

    # Default
    default = 'medium'
    if config is not None:
        default = config.get('pathway_category', {}).get('default', 'medium')
    return default


def get_feature_groups_v2(modality):
    """Get feature-level decomposition groups for v2 modalities.

    For wavelet EEG, groups are by ROI (4 groups of 40 features each).
    For inter-brain, groups are by ROI similarly.

    Args:
        modality: modality name (v2 or v1)

    Returns:
        dict mapping group_name -> (start_ch, end_ch)
    """
    if modality == 'eeg_wavelet':
        # Group by ROI: 2 components × 20 freqs per ROI = 40 features per ROI
        # Feature order: for each component, for each freq, for each ROI
        # So ROI is the innermost loop -> features [roi0, roi1, roi2, roi3] repeat
        n_freqs = 20
        n_rois = 4
        groups = {}
        from cadence.constants import EEG_ROI_NAMES
        for r_idx, roi in enumerate(EEG_ROI_NAMES):
            # Collect all column indices for this ROI
            indices = []
            for comp_idx in range(2):  # real, imag
                for f_idx in range(n_freqs):
                    col = comp_idx * (n_freqs * n_rois) + f_idx * n_rois + r_idx
                    indices.append(col)
            groups[f'wavelet_{roi}'] = indices
        return groups
    elif modality == 'eeg_interbrain':
        n_freqs = 20
        n_rois = 4
        groups = {}
        from cadence.constants import EEG_ROI_NAMES
        for r_idx, roi in enumerate(EEG_ROI_NAMES):
            indices = []
            for comp_idx in range(2):
                for f_idx in range(n_freqs):
                    col = comp_idx * (n_freqs * n_rois) + f_idx * n_rois + r_idx
                    indices.append(col)
            groups[f'interbrain_{roi}'] = indices
        return groups
    elif modality == 'blendshapes_v2':
        # 15 PCA + 15 derivatives + 1 activity
        groups = {}
        for i in range(15):
            groups[f'bl_pca_{i:02d}'] = (i, i + 1)
            groups[f'bl_pca_{i:02d}_dt'] = (15 + i, 15 + i + 1)
        groups['bl_activity'] = (30, 31)
        return groups
    elif modality == 'ecg_features_v2':
        groups = {}
        for i, name in enumerate(ECG_FEATURE_NAMES_V2):
            groups[name] = (i, i + 1)
        return groups
    else:
        # Fall back to v1
        return get_feature_groups(modality)
