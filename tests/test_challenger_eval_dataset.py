"""Tests for creative_rzero/eval/challenger/build_dataset.py — no GPU/network:
only exercises build_rows()'s deterministic pair x replicate enumeration."""

from creative_rzero.eval.challenger.build_dataset import REPLICATES_PER_PAIR, build_rows
from question_generate.creative_writing_prompts import WRITING_DOMAINS


def _total_pairs() -> int:
    return sum(len(d["subdomains"]) for d in WRITING_DOMAINS.values())


def test_build_rows_covers_every_pair_with_the_expected_replicate_count():
    rows = build_rows()

    assert len(rows) == _total_pairs() * REPLICATES_PER_PAIR

    seen = {(r["domain"], r["subdomain"], r["replicate_idx"]) for r in rows}
    assert len(seen) == len(rows)  # no duplicate (pair, replicate) slots

    for domain_key, domain in WRITING_DOMAINS.items():
        for subdomain in domain["subdomains"]:
            replicate_idxs = {
                r["replicate_idx"] for r in rows if r["domain"] == domain_key and r["subdomain"] == subdomain
            }
            assert replicate_idxs == set(range(REPLICATES_PER_PAIR))


def test_build_rows_is_reproducible_across_runs():
    a = build_rows()
    b = build_rows()

    assert a == b


def test_build_rows_replicates_sample_guidance_independently():
    # Proves the per-(domain, subdomain, replicate_idx) reseed is actually
    # varying guidance draws across replicates, not just returning the same
    # sample every time.
    rows = build_rows()
    first_domain, first_subdomain = rows[0]["domain"], rows[0]["subdomain"]
    first_pair_rows = [r for r in rows if r["domain"] == first_domain and r["subdomain"] == first_subdomain]

    guidance_variants = {tuple(r["guidance_applied"]) for r in first_pair_rows}
    assert len(guidance_variants) > 1


def test_build_rows_prompts_reflect_their_own_row_metadata():
    rows = build_rows()
    row = rows[0]

    assert row["subdomain"] in row["user_prompt"]
    assert row["domain_name"] in row["user_prompt"]
    assert "```json" in row["user_prompt"]


def test_build_rows_respects_replicates_per_pair_override():
    rows = build_rows(replicates_per_pair=2)

    assert len(rows) == _total_pairs() * 2
    replicate_idxs = {r["replicate_idx"] for r in rows}
    assert replicate_idxs == {0, 1}
