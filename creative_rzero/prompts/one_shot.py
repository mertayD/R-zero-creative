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
- "1-2": the concrete failures of a bottom-tier response to THIS task
- "3-4": what such a response still gets right, and the checkable shortfall that caps it here
- "5-6": the checkable gain over 3-4, and the specific gap that remains
- "7-8": what is now done well, and the one concrete shortfall separating it from 9-10
- "9-10": the observable behaviors of a response that meets every stated requirement

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
  could agree which band a response falls in. This applies to EVERY band:
  the middle bands (3-4, 5-6, 7-8) must each name what a response at that
  level concretely gets right and wrong about THIS task. Reusable ladder
  wording ("adequate", "competent execution", "exceptional performance")
  is a failure even in one band. Adjacent bands (3-4 vs 5-6, 5-6 vs 7-8)
  must each name at least one checkable difference; if two adjacent bands
  could describe the same response, rewrite them. And bands belong to ONE
  criterion: reusing the same band wording across two criteria is a failure.

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

The example above is illustrative ONLY: never reuse its topic, product, or
wording in your own query or criteria.

FINAL SELF-CHECK before you output the JSON:
1. Go through the refinement principles ONE BY ONE and point to the exact
   phrase in your query that realizes each; one phrase cannot satisfy two
   principles. Two principles are commonly faked when present - check them
   hardest:
   - Personalization: the query must give the WRITER an identity or lived
     experience to draw on, in second person ("As a ..., you ..." /
     "drawing on your experience of ..."). A target audience, a named
     character, or a "personalized" deliverable does NOT count.
   - "Express concisely in one sentence": the query itself must be ONE
     sentence. Count its sentences; if more than one, merge into a single
     sentence without dropping any other principle's phrase.
   A principle with no pointable phrase means the query is not done; revise it.
2. List the query's explicit requirements (word count, format, tone,
   audience, named content) and name the criterion that tests each one; a
   stated word count or format with no criterion is a failure. No criterion
   may test anything the query does not state, and no two criteria may own
   the same requirement.
3. Strip intensity words ("somewhat", "generally", "highly", "exceptionally",
   "could be more") from every band. If any two bands then read the same -
   adjacent bands of one criterion, or the same band slot of two criteria -
   rewrite them around a different checkable fact: a count, a named element
   present or missing, or a numeric range when the query states a number
   ("within 450-550 words", never "well within the word count").

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
      "1-2": "...",
      "3-4": "...",
      "5-6": "...",
      "7-8": "...",
      "9-10": "..."
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
