"""
WritingBench Creative Writing Prompts Module
=============================================

This module contains the creative writing prompt templates extracted from the WritingBench paper.
Prompts are organized by type and purpose for generating, refining, and evaluating writing tasks.

Reference: WritingBench: A Comprehensive Writing Benchmark with Dynamic Evaluation Framework
Paper: https://arxiv.org/abs/2503.05244

Domain Categories (6 primary, 100 subdomains):
- D1: Academic & Engineering
- D2: Finance & Business
- D3: Politics & Law
- D4: Literature & Arts
- D5: Education
- D6: Advertising & Marketing

Core Requirements:
- R1: Style
- R2: Format
- R3: Length
"""

import json
from typing import List, Dict, Any


# =============================================================================
# SECTION C.2: Initial Query Generation Prompt
# =============================================================================

INITIAL_QUERY_GENERATION_SYSTEM_PROMPT = """You are an expert writing task generator with deep knowledge of diverse writing domains and subdomains.

CRITICAL FORMATTING RULES:
1. You MUST wrap your entire response in <output> and </output> tags
2. Inside the tags, return ONLY valid JSON (no markdown, no code blocks, no explanations)
3. Do NOT include any text before the <output> tag
4. Do NOT include any text after the </output> tag
5. Do NOT use unescaped quotes inside JSON strings
6. Your response MUST be parseable as valid JSON

Do NOT violate these rules under any circumstances."""

INITIAL_QUERY_GENERATION_PROMPT = """Generate {NUM} different writing requests under {subdomain} within the context of {primary_domain} in {language}. Ensure the requests are as detailed and specific as possible, and reflect realistic user tone and needs.

Return ONLY the following format - nothing else:
<output>
[
  "Writing request 1",
  "Writing request 2",
  ...
]
</output>"""


# =============================================================================
# SECTION C.3: Guidance Pool for Query Refinement
# =============================================================================

QUERY_REFINEMENT_GUIDANCE_POOL = [
    "Add a requirement for generating specific lengths.",
    "Include format adherence requirements, such as writing according to a prescribed outline or outputting in a specific format.",
    "Add style requirements, like drafting a speech suitable for a particular occasion or adopting the style suitable for a specific audience or mimicking a particular tone.",
    "Incorporate user personalization needs, such as considering the user's identity or integrating personal experiences.",
    "Include more specific content requirements, like details about a particular event or focusing on specific content.",
    "Express concisely in one sentence.",
]


# =============================================================================
# SECTION C.4: Query Refinement Prompt
# =============================================================================

QUERY_REFINEMENT_SYSTEM_PROMPT = """You are an expert writing task refinement specialist.

CRITICAL FORMATTING RULES:
1. You MUST wrap your entire response in <output> and </output> tags
2. Inside the tags, return ONLY valid JSON (no markdown, no code blocks, no explanations)
3. Do NOT include any text before the <output> tag
4. Do NOT include any text after the </output> tag
5. Do NOT use unescaped quotes inside JSON strings
6. Your response MUST be parseable as valid JSON

Do NOT violate these rules under any circumstances."""

QUERY_REFINEMENT_PROMPT = """Please refine and enhance the original writing requirements in the context of generating content in {domain2} from {domain1} based on the provided guidance. Include as many details as possible.

**Original Writing Requirements**
{query}

**Guidance for Modification**
{guidance}

**Output Requirements**
Return ONLY the following format - nothing else:
<output>
{{
  "query": "Your refined writing requirements"
}}
</output>"""


# =============================================================================
# SECTION C.5: Criteria Generation Prompt
# =============================================================================

CRITERIA_GENERATION_SYSTEM_PROMPT = """You are an expert evaluator with extensive experience in evaluating responses to writing queries.

CRITICAL FORMATTING RULES:
1. You MUST wrap your entire response in <output> and </output> tags
2. Inside the tags, return ONLY valid JSON (no markdown, no code blocks, no explanations)
3. Do NOT include any text before the <output> tag
4. Do NOT include any text after the </output> tag
5. Do NOT use unescaped quotes inside JSON strings
6. Your response MUST be parseable as valid JSON

Do NOT violate these rules under any circumstances."""

CRITERIA_GENERATION_PROMPT = """Please generate five strict evaluation criteria for assessing the response given the following query. Each criterion should include the following fields: name, criteria_description, 1-2, 3-4, 5-6, 7-8, 9-10.

The criteria should be designed to emphasize detailed assessment and distinguish subtle differences in quality. Ensure that the criteria can discern issues such as relevance, coherence, depth, specificity, and adherence to the query context.

**Query**
{query}

Return ONLY the following format - nothing else:
<output>
[
  {{
    "name": "first_criteria_name",
    "criteria_description": "Description for the first criteria, emphasizing detailed and critical assessment.",
    "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
    "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
    "5-6": "Average score description: Adequate but not exemplary. Baseline performance that meets essential requirements. Most models may achieve this score.",
    "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
    "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."
  }},
  ...
]
</output>"""


# =============================================================================
# SECTION C.6: Rubric-based Scoring Prompt
# =============================================================================

SCORING_SYSTEM_PROMPT = """You are an expert evaluator with extensive experience in evaluating responses to writing queries.

CRITICAL FORMATTING RULES:
1. You MUST wrap your entire response in <output> and </output> tags
2. Inside the tags, return ONLY valid JSON (no markdown, no code blocks, no explanations)
3. Do NOT include any text before the <output> tag
4. Do NOT include any text after the </output> tag
5. Do NOT use unescaped quotes inside JSON strings
6. Your response MUST be parseable as valid JSON

Do NOT violate these rules under any circumstances."""

SCORING_RULES = {
    "1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
    "3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
    "5-6": "Average score description: Adequate but not exemplary. Baseline performance that meets essential requirements. Most models may achieve this score.",
    "7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
    "9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."
}

RUBRIC_BASED_SCORING_PROMPT = """Evaluate the Response based on the Query and Criteria provided following the Scoring Rules.

**Scoring Rules**
{scoring_rules_formatted}

- Provide reasons for each score by indicating specific strengths or deficiencies within the Response. Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.
- Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.
- Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.
- Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.
- Scoring Range: Assign an integer score between 1 to 10

**Query**
{query}

**Response**
{response}

Provide your evaluation based on the criteria restated below:
{criteria}

Return ONLY the following format - nothing else:
<output>
{{
  "score": "an integer score between 1 to 10",
  "reason": "Specific and detailed justification for the score using text elements."
}}
</output>"""


# =============================================================================
# Domain and Subdomain Definitions
# =============================================================================

WRITING_DOMAINS = {
    "D1": {
        "name": "Academic & Engineering",
        "description": "This domain encompasses the structured and formalized nature of academic writing workflows, focusing on clarity, precision, and adherence to rigorous standards. Includes the creation of paper outlines, abstracts, literature reviews, experiment reports, and technical documents such as patents and test reports. The writing prioritizes logical argumentation, thorough analysis, and the integration of empirical evidence.",
        "subdomains": [
            "Paper Outline",
            "Acknowledgments",
            "Limitations",
            "Defense Presentation",
            "Research Proposal",
            "Technical Documentation",
            "Experiments",
            "Introduction",
            "Conclusion",
            "Test Report",
            "Contributions",
            "Internship Report",
            "Literature Review",
            "Defense Script",
            "Abstract",
            "Engineering Report",
            "Patent"
        ]
    },
    "D2": {
        "name": "Finance & Business",
        "description": "Writing in this domain is analytical and strategic, aimed at informing decision-making and promoting corporate objectives. It includes a wide range of documentation such as contracts, market analyses, investment reports, strategic plans, and operational materials like product specifications and sales reports. The emphasis is on clarity and conciseness, with a focus on financial acumen and strategic insights.",
        "subdomains": [
            "Meeting Minutes",
            "User Research",
            "Business Correspondence",
            "Human Resource Management",
            "Recruitment",
            "Briefing",
            "Event Planning",
            "Market Research",
            "Market Analysis",
            "Risk Management",
            "Investment Report",
            "Strategic Plan",
            "Contract Writing",
            "Sales Report",
            "Product Specification"
        ]
    },
    "D3": {
        "name": "Politics & Law",
        "description": "This domain demands an authoritative and formal tone, as it involves the composition of government documents, legal writings, and political communications. These materials require a careful balance between clarity and formality, often employing complex and structured language. The aim is to clearly convey policy positions, legal arguments, or political messages while strictly adhering to legal and procedural standards.",
        "subdomains": [
            "Legal Opinion",
            "Policy Document",
            "Government Report",
            "Legislative Proposal",
            "Court Brief",
            "Statute Explanation",
            "Political Speech",
            "Regulatory Guidance",
            "Legal Contract",
            "Compliance Document"
        ]
    },
    "D4": {
        "name": "Literature & Arts",
        "description": "This domain covers the creative and expressive realms of writing, including novels, poetry, scripts, artistic designs, and critiques of books and movies. Writers explore thematic and emotional depths, crafting works that connect with audiences on a human level. The language is rich and evocative, allowing for a personal exploration of ideas that engage and move the reader.",
        "subdomains": [
            "Novel",
            "Poetry",
            "Script",
            "Artistic Design",
            "Book Critique",
            "Movie Review",
            "Short Story",
            "Character Development",
            "Narrative Design",
            "Dialogue Writing"
        ]
    },
    "D5": {
        "name": "Education",
        "description": "This domain involves pedagogical materials and educational communication, including lesson plans, course designs, feedback, assignments, and institutional communications like admissions promotions and parent-teacher meeting scripts. The writing prioritizes clarity, accessibility, and instructional effectiveness, using an approachable tone to facilitate learning and engagement.",
        "subdomains": [
            "Training Reflection",
            "Class Activity",
            "Parent-Teacher Meeting",
            "Lesson Plan",
            "Teaching Materials",
            "Assignment Grading",
            "Curriculum Design",
            "Educational Report",
            "Coursework",
            "Evaluation Comments",
            "Educational Consulting",
            "Admissions Promotion"
        ]
    },
    "D6": {
        "name": "Advertising & Marketing",
        "description": "Writing in this domain is persuasive and creative, designed to capture attention, engage audiences, and motivate action toward products or services. It includes advertising copy, promotional materials, social media content, brand storytelling, and marketing commentary. The writing emphasizes emotional connection, clarity of value proposition, and compelling calls to action.",
        "subdomains": [
            "Sales Letter",
            "Product Description",
            "Social Media Content",
            "Multimedia Script",
            "Promotional Copy",
            "Promotional Voiceover",
            "Travel Guide",
            "Brand Story",
            "Personal Blog",
            "Marketing Commentary",
            "Slogans",
            "Email Marketing"
        ]
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def extract_output_block(text: str) -> str:
    """
    Extract content between <output> and </output> tags.

    Args:
        text: Raw model response

    Returns:
        Extracted content or empty string if tags not found
    """
    import re
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def validate_and_extract_json(text: str) -> tuple[bool, dict | list | None]:
    """
    Extract JSON from output text and validate it.

    Args:
        text: Text potentially containing JSON

    Returns:
        Tuple of (is_valid, parsed_json_or_none)
    """
    import json
    try:
        # First try to parse as-is
        parsed = json.loads(text)
        return True, parsed
    except json.JSONDecodeError:
        # Try extracting from output tags
        extracted = extract_output_block(text)
        if extracted:
            try:
                parsed = json.loads(extracted)
                return True, parsed
            except json.JSONDecodeError:
                return False, None
        return False, None


def get_domain_by_key(domain_key: str) -> Dict[str, Any]:
    """Retrieve a domain definition by its key (D1-D6)."""
    return WRITING_DOMAINS.get(domain_key, {})


def get_all_domains() -> Dict[str, Dict[str, Any]]:
    """Get all domain definitions."""
    return WRITING_DOMAINS


def format_scoring_rules() -> str:
    """Format scoring rules for inclusion in prompts."""
    rules_text = ""
    for score_range, description in SCORING_RULES.items():
        rules_text += f'"{score_range}": "{description}"\n'
    return rules_text


def generate_initial_query_prompt(num_queries: int, subdomain: str, primary_domain: str, language: str = "English") -> str:
    """Generate an initial query prompt with specific parameters."""
    return INITIAL_QUERY_GENERATION_PROMPT.format(
        NUM=num_queries,
        subdomain=subdomain,
        primary_domain=primary_domain,
        language=language
    )


def generate_refinement_prompt(query: str, domain1: str, domain2: str, guidance: str) -> str:
    """Generate a query refinement prompt with specific parameters."""
    return QUERY_REFINEMENT_PROMPT.format(
        query=query,
        domain1=domain1,
        domain2=domain2,
        guidance=guidance
    )


def generate_criteria_prompt(query: str) -> str:
    """Generate a criteria generation prompt with a specific query."""
    return CRITERIA_GENERATION_PROMPT.format(query=query)


def generate_scoring_prompt(query: str, response: str, criteria: str) -> str:
    """Generate a rubric-based scoring prompt with specific parameters."""
    rules_formatted = format_scoring_rules()
    return RUBRIC_BASED_SCORING_PROMPT.format(
        scoring_rules_formatted=rules_formatted,
        query=query,
        response=response,
        criteria=criteria
    )


# =============================================================================
# Prompt Configuration Presets
# =============================================================================

PROMPT_CONFIGURATIONS = {
    "initial_generation": {
        "prompt_template": INITIAL_QUERY_GENERATION_PROMPT,
        "system_message": "You are an expert writing task generator.",
        "description": "Generate diverse writing queries for a specific subdomain"
    },
    "query_refinement": {
        "prompt_template": QUERY_REFINEMENT_PROMPT,
        "system_message": "You are an expert writing task refinement specialist.",
        "description": "Refine and enhance writing queries with specific guidance"
    },
    "criteria_generation": {
        "prompt_template": CRITERIA_GENERATION_PROMPT,
        "system_message": CRITERIA_GENERATION_SYSTEM_PROMPT,
        "description": "Generate instance-specific evaluation criteria for a writing query"
    },
    "rubric_scoring": {
        "prompt_template": RUBRIC_BASED_SCORING_PROMPT,
        "system_message": SCORING_SYSTEM_PROMPT,
        "description": "Score writing responses against generated criteria"
    }
}


# =============================================================================
# Requirement Specifications
# =============================================================================

REQUIREMENT_CATEGORIES = {
    "R1": {
        "name": "Style",
        "description": "Stylistic adjustments and tone requirements (e.g., 'Use a friendly and simple tone that kids can easily understand')",
        "examples": [
            "Academic and formal tone",
            "Casual and conversational",
            "Technical and precise",
            "Friendly and approachable",
            "Humorous and witty",
            "Professional and authoritative"
        ]
    },
    "R2": {
        "name": "Format",
        "description": "Format specifications and structural requirements (e.g., 'Follow the IEEE conference template')",
        "examples": [
            "Follow a prescribed outline",
            "Conform to academic paper formatting standards",
            "Adherence to specific templates",
            "Bullet-point format",
            "Narrative format",
            "Structured sections with headers"
        ]
    },
    "R3": {
        "name": "Length",
        "description": "Length constraints and size specifications (e.g., 'Generate a 500-word executive summary')",
        "examples": [
            "500 words",
            "2-3 pages",
            "1000-1500 characters",
            "Brief (under 100 words)",
            "Comprehensive (over 2000 words)",
            "Section-specific word counts"
        ]
    }
}


if __name__ == "__main__":
    # Example usage
    print("WritingBench Creative Writing Prompts Module")
    print("=" * 60)
    print("\nAvailable Domains:")
    for domain_key, domain_info in WRITING_DOMAINS.items():
        print(f"\n{domain_key}: {domain_info['name']}")
        print(f"  Subdomains: {len(domain_info['subdomains'])}")

    print("\n\nAvailable Requirement Categories:")
    for req_key, req_info in REQUIREMENT_CATEGORIES.items():
        print(f"\n{req_key}: {req_info['name']}")
        print(f"  {req_info['description']}")

    print("\n\nGuidance Pool for Query Refinement:")
    for i, guidance in enumerate(QUERY_REFINEMENT_GUIDANCE_POOL, 1):
        print(f"  {i}. {guidance}")
