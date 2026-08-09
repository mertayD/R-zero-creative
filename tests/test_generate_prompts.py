import json
import sys
import types
from types import SimpleNamespace

import pytest

from creative_rzero.config import load
from creative_rzero.data.writing_prompt import PromptBatch
from creative_rzero.paths import RunPaths
from creative_rzero.prompts.one_shot import ONE_SHOT_SYSTEM_PROMPT
from creative_rzero.steps.generate_prompts import (
    DomainSampler,
    PromptGenerationError,
    build_one_shot_prompt,
    generate_prompts,
    generate_prompts_batch,
    validate_one_shot_response,
)
from question_generate.creative_writing_prompts import QUERY_REFINEMENT_GUIDANCE_POOL, WRITING_DOMAINS

EXAMPLE_EXP = "configs/exp/example.yaml"


# =============================================================================
# DomainSampler / build_one_shot_prompt — pure, no GPU needed
# =============================================================================

def test_domain_sampler_returns_a_known_domain_and_subdomain():
    # One sampler per domain-sized seed rather than a single sample, so this
    # exercises every domain's subdomain list at least once instead of
    # whichever one seed=1 happens to land on.
    for seed in range(len(WRITING_DOMAINS)):
        domain_key, subdomain = DomainSampler(seed=seed).sample_domain_subdomain_pair()

        assert domain_key in WRITING_DOMAINS
        assert subdomain in WRITING_DOMAINS[domain_key]["subdomains"]


def test_domain_sampler_is_deterministic_for_a_fixed_seed():
    a = DomainSampler(seed=42).sample_domain_subdomain_pair()
    b = DomainSampler(seed=42).sample_domain_subdomain_pair()

    assert a == b


def test_build_one_shot_prompt_embeds_domain_and_subdomain():
    system_prompt, user_prompt, applied_guidance = build_one_shot_prompt("D1", "short story")

    # system_prompt is the fixed constant from creative_rzero.prompts.one_shot
    assert system_prompt is ONE_SHOT_SYSTEM_PROMPT
    assert "```json" in user_prompt
    assert "short story" in user_prompt
    assert WRITING_DOMAINS["D1"]["name"] in user_prompt
    assert 1 <= len(applied_guidance) <= len(QUERY_REFINEMENT_GUIDANCE_POOL)


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
    return f"```json\n{json.dumps(payload)}\n```"


def test_validate_one_shot_response_accepts_well_formed_output():
    is_valid, parsed, thinking, reason = validate_one_shot_response(_wrap(_valid_payload()))

    assert is_valid is True
    assert parsed["query"].startswith("write a detailed story")
    assert thinking == ""
    assert reason == "ok"


def test_validate_one_shot_response_strips_leading_think_block():
    text = "<think>let me plan this out</think>" + _wrap(_valid_payload())

    is_valid, parsed, thinking, reason = validate_one_shot_response(text)

    assert is_valid is True
    assert thinking == "let me plan this out"
    assert reason == "ok"


def test_validate_one_shot_response_rejects_missing_criteria():
    payload = _valid_payload()
    del payload["criteria"]

    is_valid, parsed, _, reason = validate_one_shot_response(_wrap(payload))

    assert is_valid is False
    assert parsed is None
    assert reason == "missing_query_or_criteria"


def test_validate_one_shot_response_rejects_criterion_missing_score_level():
    payload = _valid_payload()
    del payload["criteria"][0]["7-8"]

    is_valid, parsed, _, reason = validate_one_shot_response(_wrap(payload))

    assert is_valid is False
    assert reason == "criterion_missing_score_level"


def test_validate_one_shot_response_rejects_non_english_query():
    is_valid, parsed, _, reason = validate_one_shot_response(_wrap(_valid_payload(query="写一个故事")))

    assert is_valid is False
    assert reason == "non_english_query"


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
    # Per-attempt reasons: both bad responses lacked a ```json fence entirely.
    assert batch.generation_log["failure_reason_counts"] == {"missing_json_fence": 2}
    assert [f["failure_reason"] for f in batch.failures] == ["missing_json_fence"] * 2
    assert batch.failures[0]["raw_response"] == "bad"


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


def test_generate_prompts_batch_keeps_language_filter_counter_fed_via_reason():
    # The old inline is_english_output() re-check in the loop was dead code
    # (FormatValidator already rejects non-English queries) and has been
    # removed; the legacy language_filter_failures counter is now fed off
    # failure_reason == "non_english_query" instead.
    responses = [_wrap(_valid_payload(query="写一个故事")), _wrap(_valid_payload())]

    batch = generate_prompts_batch(
        _FakeLLM(responses), _FakeTokenizer(), num_prompts=1, num_format_retries=2, seed=1
    )

    assert len(batch.prompts) == 1
    assert batch.generation_log["language_filter_failures"] == 1
    assert batch.generation_log["failure_reason_counts"] == {"non_english_query": 1}


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
