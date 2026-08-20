"""creative_rzero/eval/challenger/diversity.py — near-duplicate detection
over a challenger eval run's generated queries.

Callers (run_eval.py) run this **per `(domain, subdomain)` group** — the 10
replicates built for one pair, each from a different sampled guidance_text —
rather than across the full eval set: comparing across unrelated subdomains
is low-signal (different topics are *supposed* to look different), whereas
near-identical queries *within* one pair's replicates, despite each one
having been built from different guidance, is the actual mode-collapse
signal this harness is meant to catch.

Uses TF-IDF + cosine similarity (both already repo dependencies via
scikit-learn) rather than an embedding model — cheap, dependency-free beyond
what's already vendored, and sufficient to catch near-verbatim collapse,
which is the failure mode of interest here (not subtle paraphrase detection).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SIMILARITY_THRESHOLD = 0.85


def near_duplicate_pairs(
    queries: list[str], threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Return `(i, j, similarity)` for every pair of `queries` (indices into
    the input list, i < j) whose TF-IDF cosine similarity is >= threshold.
    Returns [] for fewer than 2 queries, or if TfidfVectorizer can't build a
    vocabulary from the input (e.g. every query is empty)."""
    if len(queries) < 2:
        return []
    try:
        vectors = TfidfVectorizer().fit_transform(queries)
    except ValueError:
        return []
    sims = cosine_similarity(vectors)
    n = len(queries)
    return [
        (i, j, float(sims[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if sims[i, j] >= threshold
    ]


def duplicate_rate(queries: list[str], threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> float:
    """Fraction of `queries` involved in at least one near-duplicate pair."""
    if len(queries) < 2:
        return 0.0
    flagged: set[int] = set()
    for i, j, _ in near_duplicate_pairs(queries, threshold):
        flagged.add(i)
        flagged.add(j)
    return len(flagged) / len(queries)
