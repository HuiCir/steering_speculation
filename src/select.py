"""
Phase 2: Select — Adaptive Branch Count
========================================
DraftTarget: reads monitor_log from Generate phase,
decides how many branches to keep at each checkpoint.

Constraints:
  - High agreement (>0.7) → cap = 1 branch (don't waste compute)
  - Low agreement (<0.3) → floor = 3 branches (explore all)
  - High diversity (>0.6) → floor = 3 branches
  - Collapsed (<0.2 div) + high agree → force 1 branch
"""
import numpy as np
from .metrics import token_jaccard_distance


def decide_branch_count(agreement_rate, diversity_score):
    """
    Determine how many branches to keep.

    Args:
        agreement_rate: float [0,1], token-level agreement across branches
        diversity_score: float [0,1], hidden-state diversity

    Returns:
        branch_count: int [1,4]
        reason: str, decision rationale
    """
    if agreement_rate > 0.7:
        bc, reason = 1, "high_agree"
    elif agreement_rate > 0.5:
        bc, reason = 2, "moderate"
    elif agreement_rate > 0.3:
        bc, reason = 3, "low_agree"
    else:
        bc, reason = 4, "divergent"

    # Diversity overrides
    if diversity_score < 0.2 and agreement_rate > 0.6:
        bc = 1
        reason += "+collapsed"
    elif diversity_score > 0.6:
        bc = max(bc, 3)
        reason += "+high_div"

    return bc, reason


def select_phase(monitor_log):
    """
    Extract per-checkpoint branch count decisions.

    Returns:
        decisions: list of (token_pos, branch_count, reason, agree, div)
    """
    decisions = []
    for entry in monitor_log:
        pos, agree, div, _ = entry
        bc, reason = decide_branch_count(agree, div)
        decisions.append((pos, bc, reason, agree, div))
    return decisions


def select_diverse_branches(responses, num_to_keep):
    """
    Greedy maximum-diversity subset selection.
    Picks branches that maximize minimum pairwise Jaccard distance.

    Args:
        responses: list of N text strings
        num_to_keep: K, desired subset size

    Returns:
        selected_indices: list of K indices
    """
    if num_to_keep >= len(responses):
        return list(range(len(responses)))

    # Greedy max-min diversity selection
    selected = [0]  # Start with first branch
    for _ in range(num_to_keep - 1):
        best_idx = -1
        best_min_dist = -1
        for i in range(len(responses)):
            if i in selected:
                continue
            min_dist = min(
                token_jaccard_distance(responses[i], responses[s])
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx >= 0:
            selected.append(best_idx)

    return selected


def branch_count_summary(decisions):
    """Summarize branch count dynamics across phases."""
    if not decisions:
        return {"early": 4, "mid": 4, "late": 4}

    n = len(decisions)
    segments = {
        "early": decisions[: n // 3],
        "mid": decisions[n // 3 : 2 * n // 3],
        "late": decisions[2 * n // 3 :],
    }

    summary = {}
    for name, seg in segments.items():
        if seg:
            summary[name] = int(np.mean([d[1] for d in seg]))
        else:
            summary[name] = 4
    return summary
