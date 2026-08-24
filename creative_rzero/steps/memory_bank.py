"""steps/memory_bank.py — end-of-phase memory update for the R-Diverse
penalty (rewards/rdiverse_penalty.py).

After a challenger phase completes, re-extract every format-valid query from
the phase's rollout JSONL and append its embedding to the run family's
persistent bank, so the NEXT challenger phase's PMAP sees this phase's
output. Mirrors the paper's end-of-iteration memory update (Eq. 6); admission
is format-validity only.
"""

from __future__ import annotations

import json
import re


def _extract_query(raw: str) -> str | None:
    m = re.search(r"```(?:\s*json)?\s*(.*?)```", raw or "", re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        o = json.loads(m.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None
    q = o.get("query") if isinstance(o, dict) else None
    return q.strip() if isinstance(q, str) and q.strip() else None


def update_memory_from_phase(reward_log_path) -> int:
    """Returns the bank's new size (0 if the log is missing/empty)."""
    from creative_rzero.rewards import rdiverse_penalty

    queries = []
    try:
        with open(reward_log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("format_valid"):
                    q = _extract_query(r.get("raw_output", ""))
                    if q:
                        queries.append(q)
    except FileNotFoundError:
        print(f"[memory_bank] no reward log at {reward_log_path} — nothing to add", flush=True)
        return 0
    if not queries:
        print("[memory_bank] no valid queries in phase log — nothing to add", flush=True)
        return 0
    size = rdiverse_penalty.append_memory(queries)
    print(f"[memory_bank] +{len(queries)} queries -> bank size {size}", flush=True)
    return size
