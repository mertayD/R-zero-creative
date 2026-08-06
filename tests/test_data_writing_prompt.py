from creative_rzero.data.writing_prompt import PromptBatch, QueryRequirements, WritingPrompt


def _prompt(prompt_id="prompt_0001", domain="D1", subdomain="short story") -> WritingPrompt:
    return WritingPrompt(
        prompt_id=prompt_id,
        domain=domain,
        domain_name="Fiction",
        subdomain=subdomain,
        query="write a story",
        criteria=[{"name": "clarity", "criteria_description": "..."}],
        requirements=QueryRequirements(style="vivid", format=None, length="short"),
        guidance_applied=["show-dont-tell"],
        format_score=1,
        seed=7,
        thinking="reasoning trace",
    )


def test_writing_prompt_round_trips_through_dict():
    wp = _prompt()

    restored = WritingPrompt.from_dict(wp.to_dict())

    assert restored == wp


def test_writing_prompt_from_dict_defaults_missing_fields():
    restored = WritingPrompt.from_dict({"query": "just a query"})

    assert restored.prompt_id == ""
    assert restored.domain == ""
    assert restored.criteria == []
    assert restored.requirements == QueryRequirements()
    assert restored.format_score == 1
    assert restored.language == "English"
    assert restored.seed == 42


def test_writing_prompt_to_json_round_trips():
    import json

    wp = _prompt()
    restored = WritingPrompt.from_dict(json.loads(wp.to_json()))

    assert restored == wp


def test_query_requirements_to_dict():
    req = QueryRequirements(style="formal", format="essay", length=None)

    assert req.to_dict() == {"style": "formal", "format": "essay", "length": None}


def test_prompt_batch_add_prompt_updates_bookkeeping():
    batch = PromptBatch(batch_id="b0")

    batch.add_prompt(_prompt(prompt_id="p1", domain="D1", subdomain="short story"))
    batch.add_prompt(_prompt(prompt_id="p2", domain="D1", subdomain="short story"))
    batch.add_prompt(_prompt(prompt_id="p3", domain="D2", subdomain="poem"))
    batch.add_prompt(None)

    assert len(batch.prompts) == 3
    assert batch.domains_sampled == ["D1", "D2"]
    assert batch.subdomains_sampled == ["short story", "poem"]
    assert batch.generation_log["total_generated"] == 3
    assert batch.generation_log["skipped"] == 1
    assert batch.generation_log["total_attempted"] == 4


def test_prompt_batch_to_dict_and_json_include_prompts():
    batch = PromptBatch(batch_id="b0")
    batch.add_prompt(_prompt())

    data = batch.to_dict()

    assert data["batch_id"] == "b0"
    assert len(data["prompts"]) == 1
    assert data["prompts"][0]["query"] == "write a story"

    import json

    reloaded = json.loads(batch.to_json())
    assert reloaded == data
