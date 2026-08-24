"""R-Diverse challenger penalties (arXiv:2602.13103), text-embedding variant.

Implements Rtotal(q) = Runcertainty(q) - alpha*Prep(q, B) - beta*PMAP(q, M):

  Prep  — within-batch repetition: cluster the batch's valid queries by
          embedding cosine >= cluster_threshold (connected components, the
          embedding analog of R-Zero's BLEU agglomerative clustering); a
          query's penalty is its cluster's share of the batch, |Ci|/|B|.
  PMAP  — memory-augmented penalty vs the persistent bank M of all valid
          queries from previous challenger phases (paper Eq. 9):
            lam*[max_sim - tau_max]+ + (1-lam)*[mean_sim - tau_mean]+
          M is frozen during a phase and extended between phases
          (steps/memory_bank.py), per the paper's end-of-iteration update.

Deviation from the paper, by design: phi is a direct sentence embedding of
the query text instead of SAM's question->solver-code->code-encoder
pipeline — creative-writing queries have no canonical solver program.
The embedder is Qwen3-Embedding-4B, the same model the eval harness's
duplicate detector uses (eval/challenger/diversity.py), so training-time
penalties and eval-time measurement share one similarity scale. The
cluster threshold reuses that detector's blind-calibrated duplicate
boundary (0.72); tau_max/tau_mean are the paper's 0.5/0.25 quantile-mapped
from MiniLM's similarity distribution into Qwen's on the same blind pair
population (0.5 -> 0.47, 0.25 -> 0.20; 1,276 blind-scored within-subdomain
pairs + 3,000 cross-subdomain pairs, 2026-08-27). Memory admission is format-validity only (no
uncertainty band).

All knobs arrive via env (projected from config by verl_entry's bridge).
Embedding runs on CPU; a 32-query step batch through the 4B embedder adds
seconds per step, small next to the judge pass.
"""

from __future__ import annotations

import os
import re

import numpy as np

_embedder = None
_memory = None
_memory_loaded = False


def enabled() -> bool:
    return os.environ.get("CHALLENGER_PENALTY_ENABLED", "0") == "1"


def _cfg(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def memory_path() -> str:
    """One bank per coevolve run family, shared across its iterations."""
    exp = os.environ.get("VERL_EXPERIMENT_NAME", "default")
    family = re.sub(r"_iter\d+_.*$", "", exp)
    root = os.environ.get("STORAGE_PATH", "/storage")
    return f"{root}/memory_bank/{family}.npz"


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model = os.environ.get("CHALLENGER_PENALTY_EMBED_MODEL",
                               "Qwen/Qwen3-Embedding-4B")
        _embedder = SentenceTransformer(model, device="cpu",
                                        tokenizer_kwargs={"padding_side": "left"})
    return _embedder


def embed(queries: list[str]) -> np.ndarray:
    return _get_embedder().encode(queries, normalize_embeddings=True,
                                  show_progress_bar=False)


def _load_memory() -> np.ndarray | None:
    global _memory, _memory_loaded
    if not _memory_loaded:
        _memory_loaded = True
        path = memory_path()
        if os.path.exists(path):
            _memory = np.load(path)["emb"]
            print(f"[rdiverse] memory bank loaded: {_memory.shape[0]} entries ({path})", flush=True)
        else:
            print(f"[rdiverse] no memory bank yet ({path}) — PMAP inactive this phase", flush=True)
    return _memory


def compute_penalties(queries: list[str], batch_total: int) -> list[dict]:
    """Per valid query: {prep, p_max, p_mean, pmap, penalty}. Order-aligned
    with `queries`; `batch_total` is the full rollout batch size |B|
    (invalid rollouts count toward the denominator but join no cluster)."""
    if not queries:
        return []
    t = _cfg("CHALLENGER_PENALTY_CLUSTER_T", 0.72)
    lam = _cfg("CHALLENGER_PENALTY_LAMBDA", 0.5)
    tau_max = _cfg("CHALLENGER_PENALTY_TAU_MAX", 0.47)
    tau_mean = _cfg("CHALLENGER_PENALTY_TAU_MEAN", 0.2)
    alpha = _cfg("CHALLENGER_PENALTY_ALPHA", 1.0)
    beta = _cfg("CHALLENGER_PENALTY_BETA", 1.0)

    emb = embed(queries)
    n = len(queries)

    # Prep: connected components at cosine >= t
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sims = emb @ emb.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= t:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    csize: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        csize[r] = csize.get(r, 0) + 1

    M = _load_memory()
    out = []
    for i in range(n):
        prep = csize[find(i)] / max(1, batch_total)
        if M is not None and len(M):
            s = M @ emb[i]
            p_max, p_mean = float(s.max()), float(s.mean())
        else:
            p_max = p_mean = 0.0
        pmap = lam * max(0.0, p_max - tau_max) + (1 - lam) * max(0.0, p_mean - tau_mean)
        out.append({"prep": prep, "p_max": p_max, "p_mean": p_mean, "pmap": pmap,
                    "penalty": alpha * prep + beta * pmap})
    return out


def append_memory(queries: list[str], path: str | None = None) -> int:
    """Fold a completed phase's valid queries into the bank. Returns new size."""
    path = path or memory_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    emb = embed(queries)
    if os.path.exists(path):
        emb = np.vstack([np.load(path)["emb"], emb])
    np.savez(path, emb=emb)
    return emb.shape[0]
