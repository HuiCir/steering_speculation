"""
Phase 3: Deliver — Swarm Dispatch
==================================
Routes selected branch ensemble to downstream agents.
"""
from .metrics import jaccard_diversity, pairwise_distances


def deliver_branches(responses, mode="ensemble"):
    """
    Deliver selected branches according to mode.

    Args:
        responses: list of K text strings
        mode: str, delivery strategy
            - "ensemble": return all with diversity stats
            - "best_diverse": return top-2 most diverse
            - "single": return first response only

    Returns:
        dict with keys: responses, diversity, mode, recommendation
    """
    div = jaccard_diversity(responses) if len(responses) > 1 else 0.0

    if mode == "single" or len(responses) == 1:
        return {
            "responses": [responses[0]],
            "diversity": 0.0,
            "mode": "single",
            "recommendation": "Single answer — confidence high.",
        }

    if mode == "best_diverse" and len(responses) >= 2:
        dists = pairwise_distances(responses)
        # Find the pair with max distance
        best_pair = max(dists, key=dists.get)
        selected = [responses[best_pair[0]], responses[best_pair[1]]]
        return {
            "responses": selected,
            "diversity": dists[best_pair],
            "mode": "best_diverse",
            "recommendation": f"Top-2 most diverse strategies (dist={dists[best_pair]:.3f}).",
        }

    # Default: ensemble
    return {
        "responses": responses,
        "diversity": div,
        "mode": "ensemble",
        "recommendation": (
            f"Full ensemble of {len(responses)} branches (Jaccard div={div:.3f}). "
            "Dispatch to swarm for collective decision."
        ),
    }


def classify_branch_strategy(response):
    """Heuristic strategy classifier based on text patterns."""
    text = response.lower()
    if any(w in text for w in ["counterfactual", "adversarial", "challenge", "critical"]):
        return "adversarial"
    if any(w in text for w in ["tool results", "data shows", "output indicates", "empirical"]):
        return "tool-heavy"
    if any(w in text for w in ["creative", "unconventional", "imagine", "alternative"]):
        return "creative"
    if any(w in text for w in ["step by step", "therefore", "deduce", "logically"]):
        return "deductive"
    if any(w in text for w in ["break down", "analyze", "systematic", "structured"]):
        return "analytical"
    return "analytical"
