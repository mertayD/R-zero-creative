"""Unit tests for the R-Diverse challenger penalties — pure math, embedder
stubbed with one-hot vectors so no model download or GPU is involved."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from creative_rzero.rewards import rdiverse_penalty  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("CHALLENGER_PENALTY_ENABLED", "CHALLENGER_PENALTY_ALPHA", "CHALLENGER_PENALTY_BETA",
                "CHALLENGER_PENALTY_LAMBDA", "CHALLENGER_PENALTY_TAU_MAX",
                "CHALLENGER_PENALTY_TAU_MEAN", "CHALLENGER_PENALTY_CLUSTER_T",
                "CHALLENGER_PENALTY_EMBED_MODEL", "MEMORY_BANK_NAME"):
        monkeypatch.delenv(var, raising=False)


def _one_hot(dim, *idxs):
    out = np.zeros((len(idxs), dim), dtype=np.float32)
    for row, i in enumerate(idxs):
        out[row, i] = 1.0
    return out


def test_prep_is_cluster_share_of_full_batch(monkeypatch):
    # q0 == q1 (cluster of 2), q2 alone; batch_total=4 counts an invalid rollout
    monkeypatch.setattr(rdiverse_penalty, "embed", lambda qs: _one_hot(4, 0, 0, 1))
    monkeypatch.setattr(rdiverse_penalty, "_load_memory", lambda: None)

    pens = rdiverse_penalty.compute_penalties(["a", "a", "b"], batch_total=4)

    assert pens[0]["prep"] == pens[1]["prep"] == pytest.approx(2 / 4)
    assert pens[2]["prep"] == pytest.approx(1 / 4)
    # no memory bank yet -> PMAP inactive, penalty is alpha*prep alone
    assert all(p["pmap"] == 0.0 for p in pens)
    assert pens[0]["penalty"] == pytest.approx(0.5)


def test_pmap_hinges_on_memory_similarity(monkeypatch):
    # paper-scale taus pinned explicitly; the default (Qwen-mapped) taus are
    # locked by test_default_taus_are_qwen_mapped below
    monkeypatch.setenv("CHALLENGER_PENALTY_TAU_MAX", "0.5")
    monkeypatch.setenv("CHALLENGER_PENALTY_TAU_MEAN", "0.25")
    # one query identical to a memory entry, one orthogonal to all of memory
    monkeypatch.setattr(rdiverse_penalty, "embed", lambda qs: _one_hot(4, 0, 3))
    memory = _one_hot(4, 0, 1)  # M holds e0 and e1
    monkeypatch.setattr(rdiverse_penalty, "_load_memory", lambda: memory)

    pens = rdiverse_penalty.compute_penalties(["dup", "fresh"], batch_total=2)

    # dup: max sim 1.0, mean sim 0.5 -> 0.5*(1.0-0.5) + 0.5*(0.5-0.25) = 0.375
    assert pens[0]["p_max"] == pytest.approx(1.0)
    assert pens[0]["p_mean"] == pytest.approx(0.5)
    assert pens[0]["pmap"] == pytest.approx(0.375)
    # fresh: sims 0.0 -> both hinges clamp at zero
    assert pens[1]["pmap"] == 0.0
    # penalty = alpha*prep + beta*pmap, both singleton clusters here
    assert pens[0]["penalty"] == pytest.approx(0.5 + 0.375)
    assert pens[1]["penalty"] == pytest.approx(0.5)


def test_empty_and_disabled_paths():
    assert rdiverse_penalty.compute_penalties([], batch_total=8) == []
    assert not rdiverse_penalty.enabled()


def test_default_taus_are_qwen_mapped(monkeypatch):
    """Defaults follow the quantile-mapped Qwen scale (0.47 / 0.20)."""
    monkeypatch.setattr(rdiverse_penalty, "embed", lambda qs: _one_hot(4, 0))
    monkeypatch.setattr(rdiverse_penalty, "_load_memory", lambda: _one_hot(4, 0, 1))

    pens = rdiverse_penalty.compute_penalties(["dup"], batch_total=1)

    # max=1.0, mean=0.5 -> 0.5*(1.0-0.47) + 0.5*(0.5-0.2) = 0.415
    assert pens[0]["pmap"] == pytest.approx(0.415)


def test_step_memory_mode_grows_bank_within_phase(monkeypatch, tmp_path):
    """memory_update=step: a batch joins the bank immediately, so the next
    batch in the same phase is penalized for repeating it."""
    monkeypatch.setenv("CHALLENGER_PENALTY_MEMORY_UPDATE", "step")
    monkeypatch.setenv("CHALLENGER_PENALTY_EMBED_MODEL", "stub-embedder")
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("MEMORY_BANK_NAME", "steptest")
    monkeypatch.setattr(rdiverse_penalty, "embed", lambda qs: _one_hot(4, *([0] * len(qs))))
    monkeypatch.setattr(rdiverse_penalty, "_memory", None)
    monkeypatch.setattr(rdiverse_penalty, "_memory_loaded", False)

    first = rdiverse_penalty.compute_penalties(["q1"], batch_total=1)
    second = rdiverse_penalty.compute_penalties(["q1 again"], batch_total=1)

    assert first[0]["pmap"] == 0.0          # bank was empty when batch 1 scored
    assert second[0]["p_max"] == pytest.approx(1.0)  # batch 1 already in the bank
    assert second[0]["pmap"] > 0.0
    assert (tmp_path / "memory_bank" / "steptest.npz").exists()


def test_memory_path_is_bank_name_under_storage(monkeypatch):
    monkeypatch.setenv("MEMORY_BANK_NAME", "dup-dynamics_20260901_000000")
    monkeypatch.setenv("STORAGE_PATH", "/storage")
    assert rdiverse_penalty.memory_path() == "/storage/memory_bank/dup-dynamics_20260901_000000.npz"


def test_memory_path_refuses_to_fall_back(monkeypatch):
    monkeypatch.delenv("MEMORY_BANK_NAME", raising=False)
    with pytest.raises(RuntimeError, match="MEMORY_BANK_NAME"):
        rdiverse_penalty.memory_path()


def test_bank_from_other_embedder_is_rejected(monkeypatch, tmp_path):
    """A bank written under one embed model must not be read or appended to
    under another (the vectors share no space — the default.npz incident)."""
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("MEMORY_BANK_NAME", "mix_20260901_000000")
    monkeypatch.setattr(rdiverse_penalty, "embed", lambda qs: _one_hot(4, *([0] * len(qs))))
    monkeypatch.setattr(rdiverse_penalty, "_memory", None)
    monkeypatch.setattr(rdiverse_penalty, "_memory_loaded", False)

    monkeypatch.setenv("CHALLENGER_PENALTY_EMBED_MODEL", "model-a")
    assert rdiverse_penalty.append_memory(["q1"]) == 1

    monkeypatch.setenv("CHALLENGER_PENALTY_EMBED_MODEL", "model-b")
    with pytest.raises(RuntimeError, match="model-a"):
        rdiverse_penalty.append_memory(["q2"])
    with pytest.raises(RuntimeError, match="model-a"):
        rdiverse_penalty.check_memory_compat()


def test_untagged_legacy_bank_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("MEMORY_BANK_NAME", "legacy")
    monkeypatch.setenv("CHALLENGER_PENALTY_EMBED_MODEL", "stub-embedder")
    path = tmp_path / "memory_bank" / "legacy.npz"
    path.parent.mkdir()
    np.savez(path, emb=_one_hot(4, 0))  # pre-model-tag format
    with pytest.raises(RuntimeError, match="predates the model tag"):
        rdiverse_penalty.check_memory_compat()
