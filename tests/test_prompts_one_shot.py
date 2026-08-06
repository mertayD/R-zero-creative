from creative_rzero.prompts.one_shot import ONE_SHOT_SYSTEM_PROMPT, render_one_shot_user_prompt


def test_system_prompt_requires_output_tags():
    assert "<output>" in ONE_SHOT_SYSTEM_PROMPT
    assert "</output>" in ONE_SHOT_SYSTEM_PROMPT


def test_render_one_shot_user_prompt_embeds_all_inputs():
    rendered = render_one_shot_user_prompt(
        domain_name="Fiction",
        domain_description="Narrative writing",
        subdomain="short story",
        guidance_text="  - show, don't tell",
        language="French",
    )

    assert "Fiction" in rendered
    assert "Narrative writing" in rendered
    assert "short story" in rendered
    assert "show, don't tell" in rendered
    assert "French" in rendered
    assert "<output>" in rendered and "</output>" in rendered


def test_render_one_shot_user_prompt_defaults_to_english():
    rendered = render_one_shot_user_prompt(
        domain_name="Fiction",
        domain_description="Narrative writing",
        subdomain="short story",
        guidance_text="  (none selected)",
    )

    assert rendered.strip().endswith("Make sure outputs are in English language.")
