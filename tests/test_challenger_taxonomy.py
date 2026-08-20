"""One canonical rollout per status, shapes taken from the 4b-base audit."""

import json

from creative_rzero.challenger_taxonomy import classify

CRITERION = {"name": "Depth", "criteria_description": "d",
             "1-2": "a", "3-4": "b", "5-6": "c", "7-8": "d", "9-10": "e"}
GOOD_OBJ = {"query": "Write a formal business letter to a potential investor. " * 4,
            "criteria": [dict(CRITERION, name=f"C{i}") for i in range(5)],
            "requirements": {"style": None, "format": None, "length": None}}
GOOD_JSON = json.dumps(GOOD_OBJ, indent=2)


def test_ok_clean():
    r = classify(f"```json\n{GOOD_JSON}\n```")
    assert r["status"] == "ok_clean" and r["ok"]


def test_ok_with_artifacts_garbage_prefix():
    r = classify(f"뵉assistant\nהעבר\n```json\n{GOOD_JSON}\n```")
    assert r["status"] == "ok_with_artifacts" and r["ok"]
    assert "garbage_prefix" in r["flags"]


def test_wrapper_broken_no_fence():
    r = classify(GOOD_JSON)
    assert r["status"] == "json_ok_wrapper_broken" and not r["ok"]
    assert "no_fence" in r["flags"]


def test_wrapper_broken_decoy_fence():
    r = classify(f"Wrap your JSON: ```json\n{{........}}```\n### Output:\n```json\n{GOOD_JSON}\n```")
    assert r["status"] == "json_ok_wrapper_broken"
    assert "decoy_fence" in r["flags"]


def test_non_english_query():
    obj = dict(GOOD_OBJ, query="起草一份详细的法规指南文档，为一项新的环境政策提供指导，请详细说明。")
    r = classify(f"```json\n{json.dumps(obj, ensure_ascii=False, indent=2)}\n```")
    assert r["status"] == "json_ok_non_english"


def test_schema_wrong_query_not_string():
    obj = {"query": {"create a slogan": "for a company"}, "criteria": GOOD_OBJ["criteria"]}
    r = classify(f"```json\n{json.dumps(obj, indent=2)}\n```")
    assert r["status"] == "json_schema_wrong"
    assert "query_not_string" in r["flags"]


def test_cut_at_cap():
    partial = f"```json\n{GOOD_JSON[:400]}"
    r = classify(partial, finish_reason="length")
    assert r["status"] == "json_cut_at_cap"
    assert "hit_token_cap" in r["flags"]


def test_syntax_broken_raw_newline():
    bad = '```json\n{\n  "query": "an outline:\nI. Introduction\nII. Body",\n  "criteria": []\n}\n```'
    r = classify(bad, finish_reason="stop")
    assert r["status"] == "json_syntax_broken"


def test_degenerate_loop():
    r = classify(" lesbegirl\nWrite a lesson plan. lesbegirl\n" * 40, finish_reason="length")
    assert r["status"] == "no_json_degenerate"
    assert "repetition_loop" in r["flags"]


def test_off_task_prose():
    essay = ("This document provides a comprehensive guide for configuring a dictation "
             "system, covering both hand-written notes and voice recordings. "
             "The investigation report should follow the general report framework and "
             "conform to the language format expected by auditors. It is necessary to "
             "align oneself with the customer's viewpoint and consider what they care "
             "about when answering. Proper text indentation, callout application, and "
             "chart presentation all contribute to a document that reviewers can "
             "navigate quickly. When content is logically relevant to a previous "
             "point, indent it within the comments; otherwise apply callouts so the "
             "writing stays organized and readers are not confused by stray notes.")
    r = classify(essay, finish_reason="stop")
    assert r["status"] == "no_json_off_task"
