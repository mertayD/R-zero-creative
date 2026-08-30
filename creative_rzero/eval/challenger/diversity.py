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
scikit-learn) rather than an embedding model — cheap and dependency-free
beyond what's already vendored.

Settings calibrated against blind human-protocol judgment (2026-08-20):
1,243 within-subdomain pairs from a Qwen3-4B-Base eval run were scored 0-1
for sameness by readers blinded to all similarity values, then vectorizer
configs and thresholds were searched against those scores. Word unigrams
with English stopwords removed at threshold 0.32 gave the best F1 (0.62,
precision 0.51 / recall 0.78) for detecting true near-duplicates (blind
sameness >= 0.7, i.e. same task AND same topic); the original
no-stopword vectorizer at 0.85 flagged nothing on the same data (rank
correlation with blind judgment 0.22 vs 0.52 calibrated). Lexical methods
top out around Spearman ~0.5 on this task — an embedding-based similarity
is the known upgrade path if higher fidelity is needed.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SIMILARITY_THRESHOLD = 0.32

# Semantic layer (embedding cosine): Qwen3-Embedding-4B, calibrated 2026-08-27
# against the same blind-judgment protocol as the TF-IDF settings above
# (1,276 within-subdomain pairs, blind sameness >= 0.7 = true duplicate).
# Cosine @ 0.72 gives the best F1, 0.81 (precision 0.76 / recall 0.88), and
# Spearman 0.89 vs blind judgment; on the same pairs all-MiniLM-L6-v2 @ 0.83
# scores 0.87 / F1 0.80 and the calibrated TF-IDF above 0.52 / 0.62. Catches
# paraphrase/topic-level duplication the lexical layer structurally cannot:
# reworded same-task pairs share few content words but embed high. ~8GB in
# fp16, so run_eval frees vLLM before this loads. Requires
# sentence-transformers; lazily imported so the lexical layer keeps working
# without it.
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_SIMILARITY_THRESHOLD = 0.72

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The embedding duplicate detector needs the optional `sentence-transformers` "
                "dependency (`pip install sentence-transformers`); alternatively run the eval "
                "with `--dup-method tfidf`, which has no extra dependencies."
            ) from e
        # fp16 only where it's actually supported; CPU fallback stays fp32
        dtype = "float16" if torch.cuda.is_available() else "float32"
        _embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            model_kwargs={"torch_dtype": dtype},
            tokenizer_kwargs={"padding_side": "left"},
        )
        _embedder.max_seq_length = 2048
    return _embedder


def semantic_near_duplicate_pairs(
    queries: list[str], threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Embedding-cosine analog of near_duplicate_pairs: `(i, j, similarity)`
    for every pair at or above `threshold`."""
    if len(queries) < 2:
        return []
    emb = _get_embedder().encode(queries, normalize_embeddings=True, show_progress_bar=False)
    sims = emb @ emb.T
    n = len(queries)
    return [
        (i, j, float(sims[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if sims[i, j] >= threshold
    ]


def semantic_duplicate_rate(
    queries: list[str], threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
) -> float:
    """Fraction of `queries` involved in at least one semantic near-duplicate pair."""
    if len(queries) < 2:
        return 0.0
    flagged: set[int] = set()
    for i, j, _ in semantic_near_duplicate_pairs(queries, threshold):
        flagged.add(i)
        flagged.add(j)
    return len(flagged) / len(queries)


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
        vectors = TfidfVectorizer(stop_words="english").fit_transform(queries)
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
