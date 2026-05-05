"""Cross-session feature discovery for CADENCE v2 pipeline.

Stage 1 runs group lasso on each session independently to discover which
source features predict each target modality. Cross-session consistency
filtering then retains only features selected in a minimum number of sessions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class DiscoveryResult:
    """Result of Stage 1 group lasso discovery for one session."""

    session_name: str = ''

    # Per pathway: which source feature indices were selected
    selected_features: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)

    # Per pathway: group lasso coefficients at best lambda
    coefficients: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Per pathway: CV scores across lambda path
    cv_scores: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Per pathway: best lambda from CV
    best_lambdas: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Per pathway: number of features selected at best lambda
    n_selected: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Doubly-sparse selection metadata
    # Per pathway: stability selection scores (fraction of subsamples selecting each group)
    stability_scores: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Per pathway: block-level hit counts per feature
    block_hit_counts: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Per pathway: block-level binomial p-values per feature
    block_pvalues: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Per pathway: number of blocks analyzed
    n_blocks: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Per pathway: cluster map from pre-grouping {cluster_idx: [orig_feat_indices]}
    feature_clusters: Dict[Tuple[str, str], dict] = field(default_factory=dict)

    # Per pathway: which selection method was used
    selection_method: Dict[Tuple[str, str], str] = field(default_factory=dict)

    # Per pathway: pathway-level shift calibration p-value
    # Tests WHETHER the pathway has genuine coupling (aggregate across all features/blocks)
    block_pathway_pvalue: Dict[Tuple[str, str], float] = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    """Result of cross-session consistency filtering."""

    # Per pathway: feature indices that survived consistency filter
    consistent_features: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)

    # Per pathway: count of sessions selecting each feature
    feature_counts: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    # Total sessions analyzed
    n_sessions: int = 0

    # Minimum sessions required for consistency
    min_sessions: int = 4


def cross_session_consistency(results: List[DiscoveryResult],
                               min_sessions: int = 4,
                               total_sessions: Optional[int] = None):
    """Aggregate discovery results across sessions for consistency filtering.

    For each pathway, counts how many sessions selected each source feature.
    Features selected in >= min_sessions are retained for Stage 2.

    Args:
        results: list of DiscoveryResult from individual sessions
        min_sessions: minimum number of sessions a feature must be selected in
        total_sessions: total number of sessions (defaults to len(results))

    Returns:
        ConsistencyResult with consistent features per pathway
    """
    if total_sessions is None:
        total_sessions = len(results)

    # Collect all pathways across sessions
    all_pathways = set()
    for r in results:
        all_pathways.update(r.selected_features.keys())

    consistency = ConsistencyResult(
        n_sessions=total_sessions,
        min_sessions=min_sessions,
    )

    for pathway in all_pathways:
        # Find max feature index across all sessions for this pathway
        max_feat = 0
        for r in results:
            if pathway in r.selected_features:
                selected = r.selected_features[pathway]
                if selected:
                    max_feat = max(max_feat, max(selected) + 1)

        if max_feat == 0:
            continue

        # Count selections across sessions
        counts = np.zeros(max_feat, dtype=int)
        for r in results:
            if pathway in r.selected_features:
                for feat_idx in r.selected_features[pathway]:
                    if feat_idx < max_feat:
                        counts[feat_idx] += 1

        # Apply consistency threshold
        consistent_idx = [int(i) for i in range(max_feat)
                          if counts[i] >= min_sessions]

        consistency.consistent_features[pathway] = consistent_idx
        consistency.feature_counts[pathway] = counts

    return consistency


def build_stage2_feature_set(consistency: ConsistencyResult):
    """Extract selected column indices for Stage 2 from consistency result.

    Args:
        consistency: ConsistencyResult from cross_session_consistency()

    Returns:
        dict mapping (src_mod, tgt_mod) -> list of selected source feature indices
    """
    return {
        pathway: indices
        for pathway, indices in consistency.consistent_features.items()
        if len(indices) > 0
    }


def discovery_summary(consistency: ConsistencyResult):
    """Generate human-readable summary of discovery results.

    Returns:
        list of summary strings
    """
    lines = [
        f"Cross-session consistency: {consistency.n_sessions} sessions, "
        f"min_sessions={consistency.min_sessions}",
        "",
    ]

    for pathway in sorted(consistency.consistent_features.keys()):
        src, tgt = pathway
        features = consistency.consistent_features[pathway]
        counts = consistency.feature_counts[pathway]
        n_total = len(counts)
        n_selected = len(features)

        lines.append(f"  {src} -> {tgt}: {n_selected}/{n_total} features survived")
        if n_selected > 0 and n_selected <= 20:
            # Show individual features if not too many
            for feat_idx in features:
                lines.append(
                    f"    feature[{feat_idx}]: {counts[feat_idx]}/{consistency.n_sessions} sessions")

    return '\n'.join(lines)
