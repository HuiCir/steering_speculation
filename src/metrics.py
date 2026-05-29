"""
Diversity and Quality Metrics.
"""
import re


def token_jaccard_similarity(text_a, text_b):
    """Jaccard similarity between two texts (word-level, >=4 chars)."""
    def tok(t):
        return set(re.findall(r'\b[a-z]{4,}\b', t.lower()))
    set_a, set_b = tok(text_a), tok(text_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def token_jaccard_distance(text_a, text_b):
    """1 - Jaccard similarity."""
    return 1 - token_jaccard_similarity(text_a, text_b)


def jaccard_diversity(responses):
    """Mean pairwise Jaccard distance across all pairs."""
    dists = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            dists.append(token_jaccard_distance(responses[i], responses[j]))
    return round(sum(dists) / max(len(dists), 1), 3)


def pairwise_distances(responses):
    """All pairwise Jaccard distances as a dict."""
    result = {}
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            result[(i, j)] = token_jaccard_distance(responses[i], responses[j])
    return result


def category_diversity(responses, classifier_fn):
    """
    Number of unique strategy categories across branches.
    classifier_fn: text → category_label
    """
    cats = [classifier_fn(r) for r in responses]
    return len(set(cats)), cats


def rouge_l_score(candidate, reference):
    """Simple ROUGE-L: LCS / reference length."""
    if not reference or not candidate:
        return 0.0
    r_tok = reference.lower().split()
    c_tok = candidate.lower().split()
    m, n = len(r_tok), len(c_tok)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if r_tok[i] == c_tok[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[m][n] / m
