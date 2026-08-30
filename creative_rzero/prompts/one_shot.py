"""creative_rzero/prompts/one_shot.py — prompt copy for one-shot WritingBench
challenger-prompt generation.

Kept separate from `creative_rzero/steps/generate_prompts.py`'s sampling and
retry-loop logic (`build_one_shot_prompt` calls into this module) so the
actual prompt text is easy to find and edit without wading through control
flow, and so a future prompt-copy change diffs as a change to this file
alone.
"""

from __future__ import annotations

ONE_SHOT_SYSTEM_PROMPT = """You are an expert writing task generator with deep knowledge of diverse writing domains and subdomains.

CRITICAL FORMATTING RULES:
1. You MUST wrap your entire response in a ```json fenced code block
2. Inside the fence, return ONLY valid JSON (no explanations, no extra prose)
3. Do NOT include any text before the opening ```json fence
4. Do NOT include any text after the closing ``` fence
5. Do NOT use unescaped quotes inside JSON strings
6. Your response MUST be parseable as valid JSON

Do NOT violate these rules under any circumstances."""


def render_one_shot_user_prompt(
    domain_name: str,
    domain_description: str,
    subdomain: str,
    guidance_text: str,
    language: str = "English",
) -> str:
    """Render the one-shot generation user prompt for one domain/subdomain pair.

    `guidance_text` is the caller's pre-formatted refinement-guidance bullet
    list (or "  (none selected)") — guidance *selection* is sampling logic
    that belongs in `build_one_shot_prompt`, not in this template module.
    """
    return f"""Generate ONE detailed writing prompt for the subdomain "{subdomain}" within {domain_name}.

DOMAIN CONTEXT:
- Domain: {domain_name}
- Description: {domain_description}
- Subdomain: {subdomain}
- Language: {language}

INTERNAL REASONING STAGE (private, do not output):

STEP 1: IDEATION & CONTEXT
- Think about the {domain_name} domain and specifically the "{subdomain}" subdomain
- Consider realistic, detailed, and specific writing requests appropriate for this context
- Ensure the requests reflect the domain's standards and typical use cases

STEP 2: APPLY REFINEMENT PRINCIPLES
These refinement principles help enhance the prompt quality:
{guidance_text}

As you design the prompt, incorporate these principles to make it more specific, constrained, and valuable.

STEP 3: DESIGN EVALUATION CRITERIA
Create 5 strict evaluation criteria that can distinguish subtle differences in response quality.

Each criterion MUST include:
- name: A concise criterion name
- criteria_description: Detailed description emphasizing what the criterion evaluates
- "1-2": Critical deficiencies and major issues
- "3-4": Below average - noticeable shortcomings
- "5-6": Average - adequate but not exemplary
- "7-8": Above average - competent execution
- "9-10": High - exceptional performance

Every criterion must:
- Grade the writer's RESPONSE, never the prompt itself. Bands like "the prompt
  clearly states..." are wrong; bands describe what a submitted response does.
- Be specific to THIS task: name and description reference its actual subject,
  audience, format, or constraints, not reusable quality words.
- Only test what the query states. If the query never gives a word count, no
  criterion may grade word count.
- Be fulfillable in plain text (no fonts, colors, LaTeX rendering) and never
  reward invented citations, statistics, or sources.
- Have bands that describe OBSERVABLY different responses, so two readers
  could agree which band a response falls in.

EXAMPLES - for an unrelated task ("Write a 300-word product description for a
mechanical keyboard aimed at first-time buyers"):

GOOD criterion:
  name: "Jargon accessibility for first-time buyers"
  criteria_description: "Whether switch types, actuation force, and keycap
    terms are introduced with plain-language explanations a newcomer can use."
  1-2: "Unexplained jargon throughout (e.g. 'tactile 55g Zilents' with no
    gloss); a newcomer cannot follow the key selling points."
  5-6: "Most terms glossed, but at least one purchase-relevant spec is left
    unexplained or explained inaccurately."
  9-10: "Every technical term is introduced in plain language on first use,
    and each explanation ties back to what a first-time buyer would feel or
    notice."

BAD criterion (do NOT produce these):
  name: "Relevance" / description: "How relevant the response is to the task"
    -> generic; fits any writing task, names no task-specific content.
  1-2: "The prompt fails to specify the target audience"
    -> grades the prompt, not the response.
  9-10: "Cites at least 5 peer-reviewed sources"
    -> the query never asked for sources; rewards fabrication.

STEP 4: IDENTIFY REQUIREMENTS
Look for and identify any:
- Style requirements (e.g., formal, casual, tone, audience)
- Format requirements (e.g., structure, template, outline)
- Length requirements (e.g., word count, page count, character limits)

OUTPUT STAGE - Return ONLY this JSON format wrapped in a ```json fenced code block:
```json
{{
  "query": "Your detailed, polished, specific writing prompt that reflects the refinement principles",
  "criteria": [
    {{
      "name": "Criterion 1 Name",
      "criteria_description": "Detailed description for the first criteria, emphasizing detailed and critical assessment.",
      "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
      "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
      "5-6": "Average score description: Adequate but not exemplary. Baseline performance that meets essential requirements.",
      "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed.",
      "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness."
    }},
    {{
      "name": "Criterion 2 Name",
      "criteria_description": "Detailed description for the second criteria...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }},
    {{
      "name": "Criterion 3 Name",
      "criteria_description": "...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }},
    {{
      "name": "Criterion 4 Name",
      "criteria_description": "...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }},
    {{
      "name": "Criterion 5 Name",
      "criteria_description": "...",
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
    }}
  ],
  "requirements": {{
    "style": "Style requirement if explicitly mentioned, null otherwise",
    "format": "Format requirement if explicitly mentioned, null otherwise",
    "length": "Length requirement if explicitly mentioned, null otherwise"
  }}
}}
```

CRITICAL REMINDERS:
- Do NOT output your internal reasoning. Only output the JSON block.
- Use a ```json fenced code block to wrap the JSON output.
- Ensure the query is specific and detailed, reflecting the refinement principles you applied.
- Each 5 criterion must have all 5 score levels (1-2, 3-4, 5-6, 7-8, 9-10) with detailed descriptions.
- Be strict in criteria design to distinguish subtle differences in quality.
- Criteria grade the response, never the prompt; follow the GOOD example's specificity.
- Reference exact aspects of what makes responses succeed or fail at each level.
- Make sure outputs are in {language} language."""
