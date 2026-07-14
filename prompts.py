"""
prompts.py
All LLM prompt templates live here, separated from the calling code, so
prompt iterations are easy to diff/review independently of pipeline logic.
"""

EXTRACTION_SYSTEM_PROMPT = """You are a meticulous legal contract analyst. You extract clauses \
verbatim from contracts -- you never paraphrase, invent, or infer clause text that is not \
actually present in the document. If a clause type is genuinely absent, you say so explicitly \
rather than guessing."""

EXTRACTION_TASK_INSTRUCTIONS = """Read the contract text below and extract these three clause \
types:

1. termination_clause -- conditions under which either party may terminate the agreement \
(e.g. for convenience, for cause, notice periods).
2. confidentiality_clause -- obligations to keep information confidential, non-disclosure terms, \
exceptions to confidentiality.
3. liability_clause -- limitation of liability, indemnification, or liability cap language.

For each clause type, return:
- "found": true or false
- "text": the exact verbatim quoted text of the clause from the contract (empty string if not found). \
Keep this to the most relevant 1-3 sentences -- do not paste entire sections.
- "confidence": "high", "medium", or "low", reflecting how clearly this maps to the requested clause type.

Respond using the provided JSON tool schema only.

CONTRACT TEXT:
{contract_text}
"""

# Few-shot exemplars (bonus feature). Populate at runtime from CUAD's
# master_clauses.csv ground truth via prompts.build_few_shot_block().
FEW_SHOT_HEADER = "Here are examples of correctly extracted clauses from other contracts:\n"


def build_few_shot_block(examples: list) -> str:
    """
    examples: list of dicts like
        {"clause_type": "confidentiality_clause", "example_text": "..."}
    Returns a formatted string to prepend to the extraction prompt.
    """
    if not examples:
        return ""
    lines = [FEW_SHOT_HEADER]
    for ex in examples:
        lines.append(f'- [{ex["clause_type"]}]: "{ex["example_text"]}"')
    return "\n".join(lines) + "\n\n"


SUMMARY_SYSTEM_PROMPT = """You are a legal analyst who writes clear, neutral, concise contract \
summaries for a non-lawyer business audience."""

SUMMARY_TASK_INSTRUCTIONS = """Write a summary of the following contract in 100-150 words. Cover:
- The purpose of the agreement.
- The key obligations of each party.
- Notable risks, penalties, or liability exposure.

Be concise and neutral. Do not add a preamble like "This contract is about" -- start directly \
with the substance. Do not exceed 150 words.

CONTRACT TEXT:
{contract_text}
"""

# Tool schema forcing structured JSON output for extraction (see extractor.py).
EXTRACTION_TOOL_SCHEMA = {
    "name": "record_clause_extraction",
    "description": "Record the extracted clauses found in a contract.",
    "input_schema": {
        "type": "object",
        "properties": {
            clause: {
                "type": "object",
                "properties": {
                    "found": {"type": "boolean"},
                    "text": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["found", "text", "confidence"],
            }
            for clause in ["termination_clause", "confidentiality_clause", "liability_clause"]
        },
        "required": ["termination_clause", "confidentiality_clause", "liability_clause"],
    },
}
