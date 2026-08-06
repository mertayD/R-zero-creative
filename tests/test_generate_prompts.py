import json
import sys
import types
from types import SimpleNamespace

import pytest

from creative_rzero.config import load
from creative_rzero.data.writing_prompt import PromptBatch
from creative_rzero.paths import RunPaths
from creative_rzero.steps.generate_prompts import (
    DomainSampler,
    PromptGenerationError,
    build_one_shot_prompt,
    generate_prompts,
    generate_prompts_batch,
    validate_one_shot_response,
)
from question_generate.creative_writing_prompts import WRITING_DOMAINS

EXAMPLE_EXP = "configs/exp/example.yaml"


# =============================================================================
# DomainSampler / build_one_shot_prompt — pure, no GPU needed
# =============================================================================

def test_domain_sampler_returns_a_known_domain_and_subdomain():
    sampler = DomainSampler(seed=1)
    domain_key, subdomain = sampler.sample_domain_subdomain_pair()

    assert domain_key in WRITING_DOMAINS
    assert subdomain in WRITING_DOMAINS[domain_key]["subdomains"]


def test_domain_sampler_is_deterministic_for_a_fixed_seed():
    a = DomainSampler(seed=42).sample_domain_subdomain_pair()
    b = DomainSampler(seed=42).sample_domain_subdomain_pair()

    assert a == b


def test_build_one_shot_prompt_embeds_domain_and_subdomain():
    system_prompt, user_prompt, applied_guidance = build_one_shot_prompt("D1", "short story")

    assert "<output>" in system_prompt or "<output>" in user_prompt
    assert "short story" in user_prompt
    assert WRITING_DOMAINS["D1"]["name"] in user_prompt
    assert 1 <= len(applied_guidance) <= len(
        __import__("question_generate.creative_writing_prompts", fromlist=["QUERY_REFINEMENT_GUIDANCE_POOL"]).QUERY_REFINEMENT_GUIDANCE_POOL
    )


# =============================================================================
# validate_one_shot_response
# =============================================================================

def _valid_payload(query="write a detailed story about a lighthouse keeper"):
    levels = {lvl: "..." for lvl in ("1-2", "3-4", "5-6", "7-8", "9-10")}
    criterion = {"name": "clarity", "criteria_description": "how clear it is", **levels}
    return {
        "query": query,
        "criteria": [criterion],
        "requirements": {"style": None, "format": None, "length": None},
    }


def _wrap(payload: dict) -> str:
    return f"<output>{json.dumps(payload)}</output>"


def test_validate_one_shot_response_accepts_well_formed_output():
    is_valid, parsed, thinking = validate_one_shot_response(_wrap(_valid_payload()))

    assert is_valid is True
    assert parsed["query"].startswith("write a detailed story")
    assert thinking == ""


def test_validate_one_shot_response_strips_leading_think_block():
    text = "<think>let me plan this out</think>" + _wrap(_valid_payload())

    is_valid, parsed, thinking = validate_one_shot_response(text)

    assert is_valid is True
    assert thinking == "let me plan this out"


def test_validate_one_shot_response_rejects_missing_criteria():
    payload = _valid_payload()
    del payload["criteria"]

    is_valid, parsed, _ = validate_one_shot_response(_wrap(payload))

    assert is_valid is False
    assert parsed is None


def test_validate_one_shot_response_rejects_criterion_missing_score_level():
    payload = _valid_payload()
    del payload["criteria"][0]["7-8"]

    is_valid, parsed, _ = validate_one_shot_response(_wrap(payload))

    assert is_valid is False


def test_validate_one_shot_response_rejects_non_english_query():
    is_valid, parsed, _ = validate_one_shot_response(_wrap(_valid_payload(query="写一个故事")))

    assert is_valid is False


# =============================================================================
# generate_prompts_batch — the retry/validation control flow. Runs without a
# GPU by injecting a fake `vllm` module (only its SamplingParams constructor
# is touched) and a fake model/tokenizer.
# =============================================================================

class _FakeTokenizer:
    chat_template = None  # forces the plain "system: ...\n\nuser: ..." branch
    eos_token_id = 0


class _FakeLLM:
    """Returns the next canned response text on each .generate() call."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def generate(self, prompts, sampling_params=None):
        text = next(self._responses)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]


@pytest.fixture(autouse=True)
def fake_vllm_module(monkeypatch):
    fake_module = types.ModuleType("vllm")
    fake_module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_module)
    yield


def test_generate_prompts_batch_collects_valid_prompts():
    responses = [_wrap(_valid_payload()) for _ in range(3)]

    batch = generate_prompts_batch(_FakeLLM(responses), _FakeTokenizer(), num_prompts=3, seed=1)

    assert len(batch.prompts) == 3
    assert batch.generation_log["format_validation_failures"] == 0


def test_generate_prompts_batch_retries_then_succeeds_within_a_slot():
    # First slot: 1 bad response then 1 good one (within num_format_retries=2).
    responses = ["not valid at all", _wrap(_valid_payload())]

    batch = generate_prompts_batch(
        _FakeLLM(responses), _FakeTokenizer(), num_prompts=1, num_format_retries=2, seed=1
    )

    assert len(batch.prompts) == 1
    assert batch.generation_log["format_validation_failures"] == 0
    # generation_log["total_attempted"]/["skipped"] only move via
    # PromptBatch.add_prompt(), which this loop calls once per *successful*
    # slot and explicitly never with None on failure (see the "Don't call
    # batch.add_prompt(None)" comment in generate_prompts_batch) — so this
    # counts slots added, not raw generate() calls or retries.
    assert batch.generation_log["total_attempted"] == 1


def test_generate_prompts_batch_gives_up_on_a_slot_after_max_retries():
    # First slot exhausts both retries, second slot succeeds immediately.
    responses = ["bad", "still bad", _wrap(_valid_payload())]

    batch = generate_prompts_batch(
        _FakeLLM(responses), _FakeTokenizer(), num_prompts=1, num_format_retries=2, seed=1
    )

    assert len(batch.prompts) == 1
    assert batch.generation_log["format_validation_failures"] == 1
    # The exhausted slot never calls add_prompt, so it contributes 0 to
    # total_attempted — only the slot that eventually succeeded does.
    assert batch.generation_log["total_attempted"] == 1


def test_generate_prompts_batch_stops_at_attempt_cap_when_model_never_succeeds():
    # max_total_attempts = num_prompts * (num_format_retries + 2) = 2 * 4 = 8
    responses = ["bad"] * 20

    batch = generate_prompts_batch(
        _FakeLLM(responses), _FakeTokenizer(), num_prompts=2, num_format_retries=2, seed=1
    )

    assert len(batch.prompts) == 0
    assert batch.generation_log["format_validation_failures"] == 4
    # No slot ever succeeded, so add_prompt was never called — total_attempted
    # stays 0 even though 4 slots (8 raw generate() calls) were tried. This is
    # the same "total_attempted only tracks successes" behavior as above; it
    # means generation_log["total_attempted"]/["skipped"] cannot currently
    # answer "how many attempts/skips happened", only format_validation_failures
    # and language_filter_failures can. Worth relying on those two, not these.
    assert batch.generation_log["total_attempted"] == 0


# NOTE: generate_prompts_batch's own inline check
# `if is_valid and not is_english_output(parsed_json.get('query', ''))` (and
# the language_filter_failures counter it feeds) is unreachable in practice:
# validate_one_shot_response -> FormatValidator.validate_response already
# rejects a non-English `query` and returns is_valid=False before that second
# check ever runs — see test_validate_one_shot_response_rejects_non_english_query
# above, which exercises the same input at the layer that actually catches it.
# Flagging here rather than deleting the dead branch — not touching behavior
# beyond the file consolidation.


# =============================================================================
# generate_prompts() — the typed pipeline step, with run_generation_fn faked
# out (never touches vllm/transformers).
# =============================================================================

@pytest.fixture
def paths(tmp_path) -> RunPaths:
    return RunPaths(tmp_path, "example", "20260805_120000", iteration=1)


def _fake_batch(n: int) -> PromptBatch:
    batch = PromptBatch(batch_id="b0")
    for i in range(n):
        batch.generation_log["total_generated"] = n
        batch.generation_log["total_attempted"] = n
    batch.prompts = [object()] * n  # generate_prompts() only checks len(batch.prompts)
    return batch


def test_generate_prompts_calls_run_generation_with_expected_args_and_returns_path(paths):
    cfg = load(EXAMPLE_EXP, cli_args=["solver.num_train=3", "solver.num_val=1"])
    captured = {}

    def fake_run_generation(**kwargs):
        captured.update(kwargs)
        return _fake_batch(4)

    result = generate_prompts(
        cfg, paths, "/storage/models/challenger/huggingface", run_generation_fn=fake_run_generation
    )

    assert result == paths.prompts_json(suffix=cfg.run.profile)
    assert captured["model"] == "/storage/models/challenger/huggingface"
    assert captured["num_samples"] == 4
    assert captured["seed"] == cfg.run.seed
    assert captured["out_path"] == paths.prompts_json(suffix=cfg.run.profile)
    assert captured["save_name"] == f"{paths.iter_abbr}_solver_prompts"
    assert captured["suffix"] == cfg.run.profile


def test_generate_prompts_raises_on_shortfall(paths):
    cfg = load(EXAMPLE_EXP, cli_args=["solver.num_train=3", "solver.num_val=1"])

    with pytest.raises(PromptGenerationError, match="expected >= 4 prompts, got 2"):
        generate_prompts(
            cfg, paths, "ckpt", run_generation_fn=lambda **kwargs: _fake_batch(2)
        )
