"""creative_rzero/challenger_taxonomy.py — fine-grained challenger rollout taxonomy.

The training-path validator (utils.FormatValidator) answers one question: does
the first fenced block parse? That collapses very different behaviors into
challenger_format_invalid. This module classifies WHAT the rollout actually is,
from a manual audit of every failure in run 4b-base_20260816_005953 (the
resulting checker agreed with FormatValidator's ok/fail boundary on 160/160
rollouts while sub-typing the failures).

Each rollout gets exactly ONE status:

  ok_clean               one fenced block, valid JSON, correct schema, English,
                         no junk anywhere
  ok_with_artifacts      extraction still succeeds, but junk surrounds the valid
                         JSON (garbage prefix, leaked role token, extra fence)
  json_ok_wrapper_broken complete valid schema-correct JSON exists, but the
                         fencing is broken (missing, unclosed, or a decoy fence
                         earlier that breaks a first-fence parser)
  json_ok_non_english    complete valid JSON, query not in English
  json_schema_wrong      parses, wrong shape (query not a string, <5 criteria,
                         missing score bands)
  json_cut_at_cap        genuine JSON attempt cut off mid-object by the token cap
  json_syntax_broken     JSON attempt, unparseable (raw newlines in strings, bad
                         commas, " + " concat), with budget left
  no_json_degenerate     no JSON attempt; loops, token salad, or (near-)empty
  no_json_off_task       no JSON attempt; coherent text doing the wrong thing
                         (essay, fake chat, answered the task instead of writing
                         a prompt)

The first two count as OK. Independently of status, FLAGS mark each observable
phenomenon (several usually co-occur): garbage_prefix, role_token_leak,
repetition_loop, near_empty, hit_token_cap, wrong_language_text,
prompt_template_echo, preamble_before_json, decoy_fence, fence_unclosed,
multiple_json_fences, no_fence, trailing_text_after_json, non_english_query,
criterion_fragments_present, plus schema detail flags.

classify() is deterministic and dependency-light (stdlib only). When rollouts
are generated under data.response_prefill, prepend the prefill before calling —
same contract as the reward caller.
"""

from __future__ import annotations

import json
import re
import unicodedata

CAP_DEFAULT = 2048

CRITERION_KEYS = {"name", "criteria_description", "1-2", "3-4", "5-6", "7-8", "9-10"}
SCORE_BAND_RE = re.compile(r'"(?:1-2|3-4|5-6|7-8|9-10)"\s*:')
TEMPLATE_MARKERS = ("INTERNAL REASONING STAGE", "OUTPUT STAGE - Return ONLY",
                    "CRITICAL REMINDERS", "DESIGN EVALUATION CRITERIA")
ROLE_LEAK_RE = re.compile(
    r"<\|im_(?:start|end)\|>"
    r"|</?\s*(?:user|assistant|system|使用者|用户)\s*>"
    r"|^\s*(?:assistant|user|system)\s*$"
    r"|^(?:Human|Assistant|User)\s*:", re.MULTILINE | re.IGNORECASE)
NONLATIN_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "HEBREW",
                    "CYRILLIC", "THAI", "DEVANAGARI", "ETHIOPIC", "TAMIL",
                    "GREEK", "ARMENIAN", "GEORGIAN", "BENGALI", "MYANMAR", "KHMER", "LAO")


def _char_script(ch):
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"


def _nonlatin_letters(text):
    lat = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        s = _char_script(ch)
        if s == "LATIN":
            lat += 1
        elif any(s.startswith(p) for p in NONLATIN_SCRIPTS):
            other += 1
    return lat, other


def find_json_objects(text):
    """All brace-balanced spans that json.loads to a dict: [(start, end, obj)]."""
    out, i = [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        depth, j, instr, esc = 0, i, False, False
        while j < len(text):
            ch = text[j]
            if esc:
                esc = False
            elif ch == "\\" and instr:
                esc = True
            elif ch == '"':
                instr = not instr
            elif not instr:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if j < len(text) and depth == 0:
            try:
                o = json.loads(text[i:j + 1])
                if isinstance(o, dict):
                    out.append((i, j + 1, o))
                    i = j + 1
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
        i += 1


def schema_check(obj):
    """Return list of schema flags; empty list = schema-complete."""
    flags = []
    q = obj.get("query")
    if not isinstance(q, str) or not q.strip():
        flags.append("query_not_string" if q is not None else "query_missing")
    crit = obj.get("criteria")
    if not isinstance(crit, list):
        flags.append("criteria_missing")
    else:
        if len(crit) < 5:
            flags.append("criteria_count_wrong")
        bad = sum(1 for c in crit
                  if not isinstance(c, dict) or not CRITERION_KEYS <= set(c))
        if bad:
            flags.append("criterion_keys_missing")
    if not isinstance(obj.get("requirements"), dict):
        flags.append("requirements_missing")  # advisory only, never fails status
    return flags


def query_english(obj):
    q = obj.get("query")
    if not isinstance(q, str):
        return True
    lat, other = _nonlatin_letters(q)
    return other == 0 or other / max(1, lat + other) < 0.02


def repetition_loop(text):
    """Any 10-80 char unit repeated >=5 times consecutively, or any 40-char
    chunk recurring >=8 times anywhere, or one word >=30 times consecutively."""
    if re.search(r"(.{10,80}?)(?:\1){4,}", text, re.DOTALL):
        return True
    for m in range(0, max(0, len(text) - 40), 200):
        chunk = text[m:m + 40]
        if chunk.strip() and text.count(chunk) >= 8:
            return True
    return bool(re.search(r"\b(\S{2,20})(?:\s+\1){29,}\b", text))


def classify(raw, finish_reason=None, n_tokens=None, cap=CAP_DEFAULT):
    """Classify one challenger rollout. Returns {"status", "ok", "flags"}.

    finish_reason ("length" = hit the cap) or n_tokens enable the
    json_cut_at_cap status; without either, cap hits can't be distinguished
    from syntax breaks.
    """
    text = raw or ""
    stripped = text.strip()
    flags = set()

    fences = [m.start() for m in re.finditer(r"```", text)]
    json_fences = [m.start() for m in re.finditer(r"```\s*json", text, re.IGNORECASE)]
    first_block = re.search(r"```(?:\s*json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    objs = find_json_objects(text)
    complete = [(a, b, o) for a, b, o in objs
                if not [f for f in schema_check(o) if f != "requirements_missing"]]

    cap_hit = (finish_reason == "length") if finish_reason is not None else (
        n_tokens is not None and n_tokens >= cap - 8)
    if cap_hit:
        flags.add("hit_token_cap")

    # a "JSON attempt" = fence, or an object-ish span mentioning the contract keys
    attempt = bool(json_fences) or bool(objs) or (
        '"query"' in text and text.find("{") >= 0) or len(SCORE_BAND_RE.findall(text)) >= 2

    content_start = min([a for a, _, _ in objs] + fences + [len(text)])
    pre = text[:content_start].strip()
    if pre:
        lat, other = _nonlatin_letters(pre)
        if (other > 0 or "�" in pre or "{{" in pre
                or any(unicodedata.category(c) == "So" for c in pre)):
            flags.add("garbage_prefix")
        elif len(pre) >= 40:
            flags.add("preamble_before_json")
    if ROLE_LEAK_RE.search(text):
        flags.add("role_token_leak")
    outside = text
    for a, b, _ in reversed(complete):  # don't let valid rubric boilerplate trip the loop detector
        outside = outside[:a] + outside[b:]
    if repetition_loop(outside):
        flags.add("repetition_loop")
    if len(stripped) < 200:
        flags.add("near_empty")
    lat_all, other_all = _nonlatin_letters(text)
    if other_all / max(1, lat_all + other_all) > 0.3:
        flags.add("wrong_language_text")
    if any(mk in text for mk in TEMPLATE_MARKERS):
        flags.add("prompt_template_echo")
    if len(fences) % 2 == 1:
        flags.add("fence_unclosed")
    if len(json_fences) >= 2:
        flags.add("multiple_json_fences")
    if not fences and objs:
        flags.add("no_fence")

    first_block_parses = False
    if first_block:
        try:
            first_block_parses = isinstance(json.loads(first_block.group(1).strip()), dict)
        except (json.JSONDecodeError, ValueError):
            pass
    if fences and complete and not first_block_parses:
        flags.add("decoy_fence")
    if complete and fences:
        a, b, _ = complete[-1]
        close = text.find("```", b)
        tail = text[close + 3:] if close >= 0 else text[b:]
        if len(tail.strip()) > 20:
            flags.add("trailing_text_after_json")

    # unterminated brace structure at end of text = the attempt itself was cut
    tail_open = 0
    instr = esc = False
    for ch in text:
        if esc:
            esc = False
        elif ch == "\\" and instr:
            esc = True
        elif ch == '"':
            instr = not instr
        elif not instr:
            if ch == "{":
                tail_open += 1
            elif ch == "}":
                tail_open = max(0, tail_open - 1)
    ends_unterminated = tail_open > 0 or instr

    # only TOP-LEVEL contract attempts count for schema verdicts; a lone
    # criterion dict is a fragment of a broken outer object, not an attempt
    contract_objs = [(a, b, o) for a, b, o in objs if "query" in o or "criteria" in o]
    criterion_frags = [o for *_, o in objs
                       if CRITERION_KEYS & set(o.keys())
                       and "query" not in o and "criteria" not in o]

    if complete:
        a, b, obj = complete[-1]
        properly_fenced = (first_block is not None and first_block_parses
                          and first_block.start() <= a and "decoy_fence" not in flags)
        artifact_flags = flags & {"garbage_prefix", "preamble_before_json",
                                  "trailing_text_after_json", "repetition_loop",
                                  "role_token_leak", "multiple_json_fences",
                                  "prompt_template_echo"}
        if not query_english(obj):
            status = "json_ok_non_english"
            flags.add("non_english_query")
        elif properly_fenced:
            status = "ok_with_artifacts" if artifact_flags else "ok_clean"
        else:
            status = "json_ok_wrapper_broken"
    elif attempt and cap_hit and ends_unterminated:
        status = "json_cut_at_cap"
    elif contract_objs:
        status = "json_schema_wrong"
        for *_, o in contract_objs:
            flags.update(f for f in schema_check(o) if f != "requirements_missing")
    elif attempt:
        status = "json_syntax_broken"
        if criterion_frags:
            flags.add("criterion_fragments_present")
    elif flags & {"repetition_loop", "near_empty", "wrong_language_text"}:
        status = "no_json_degenerate"
    else:
        status = "no_json_off_task"

    return {"status": status, "ok": status in ("ok_clean", "ok_with_artifacts"),
            "flags": sorted(flags)}
