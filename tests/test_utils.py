import json

from creative_rzero.utils import FormatValidator, find_cjk_matches, is_english_output


def test_is_english_output_true_for_plain_english():
    assert is_english_output("a perfectly ordinary sentence.") is True


def test_is_english_output_false_when_cjk_present():
    assert is_english_output("hello 世界") is False


def test_find_cjk_matches_returns_each_cjk_char():
    assert find_cjk_matches("a世b界c") == ["世", "界"]


def test_find_cjk_matches_empty_for_pure_english():
    assert find_cjk_matches("nothing to see here") == []


def _wrap(payload: dict) -> str:
    return f"```json\n{json.dumps(payload)}\n```"


def test_validate_json_fence_extracts_inner_text():
    ok, extracted = FormatValidator.validate_json_fence('noise ```json\n{"a": 1}\n``` trailing')
    assert ok is True
    assert extracted == '{"a": 1}'


def test_validate_json_fence_accepts_bare_fence_without_language_tag():
    ok, extracted = FormatValidator.validate_json_fence('```\n{"a": 1}\n```')
    assert ok is True
    assert extracted == '{"a": 1}'


def test_validate_json_fence_missing_fence():
    ok, extracted = FormatValidator.validate_json_fence("no fence here")
    assert ok is False
    assert extracted == ""


def test_validate_json_valid_and_invalid():
    assert FormatValidator.validate_json('{"a": 1}') == (True, {"a": 1})
    ok, parsed = FormatValidator.validate_json("{not json")
    assert ok is False
    assert parsed is None


def test_validate_response_valid_dict_with_string_query():
    score, parsed, reason = FormatValidator.validate_response(_wrap({"query": "write a story"}))
    assert score == 1
    assert parsed == {"query": "write a story"}
    assert reason == "ok"


def test_validate_response_missing_fence_is_invalid():
    score, parsed, reason = FormatValidator.validate_response("no json fence")
    assert score == -1
    assert parsed is None
    assert reason == "missing_json_fence"


def test_validate_response_invalid_json_is_invalid():
    score, parsed, reason = FormatValidator.validate_response("```json\nnot json\n```")
    assert score == -1
    assert parsed is None
    assert reason == "invalid_json"


def test_validate_response_non_dict_top_level_is_invalid():
    score, parsed, reason = FormatValidator.validate_response(_wrap(["a", "list"]))
    assert score == -1
    assert parsed is None
    assert reason == "top_level_not_dict"


def test_validate_response_non_string_query_is_invalid():
    score, parsed, reason = FormatValidator.validate_response(_wrap({"query": {"nested": True}}))
    assert score == -1
    assert parsed is None
    assert reason == "query_not_string"


def test_validate_response_non_english_query_is_invalid():
    score, parsed, reason = FormatValidator.validate_response(_wrap({"query": "写一个故事"}))
    assert score == -1
    assert parsed is None
    assert reason == "non_english_query"


def test_validate_response_allows_missing_query_field():
    # query is optional at this layer — validate_one_shot_response (a stricter,
    # generation-specific check) is what requires it.
    score, parsed, reason = FormatValidator.validate_response(_wrap({"other": "field"}))
    assert score == 1
    assert parsed == {"other": "field"}
    assert reason == "ok"
