"""
Generates a 100-150 word summary of a contract using the configured LLM
provider (Anthropic or OpenAI, see config.LLM_PROVIDER).

For long contracts, we ground the summary using the already-extracted
clauses (from extractor.py) plus a truncated head of the contract text,
rather than re-chunking and summarizing chunk-by-chunk -- a summary is a
holistic judgment and doesn't merge well across independently-summarized
chunks the way clause extraction does.
"""

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from config import MAX_TOKENS_SUMMARY, MAX_CONTEXT_TOKENS
from prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_TASK_INSTRUCTIONS
from src.chunker import needs_chunking
from src.llm_client import call_text

logger = logging.getLogger(__name__)


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Cheap head-truncation fallback for contracts too long to summarize whole."""
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def summarize_contract(contract_text: str, extracted_clauses: dict = None) -> str:
    """
    Generate a 100-150 word summary of the contract.
    If the contract is too long for a single call, it is truncated to the
    context budget and the already-extracted clauses (if provided) are
    appended as grounding context so key risk/obligation details aren't lost.
    """
    text_for_summary = contract_text
    if needs_chunking(contract_text):
        logger.info("Contract exceeds context budget, truncating for summary and grounding with extracted clauses.")
        text_for_summary = _truncate_to_token_budget(contract_text, MAX_CONTEXT_TOKENS - 500)

        if extracted_clauses:
            clause_context = "\n\nKEY CLAUSES ALREADY IDENTIFIED (use these to inform the summary):\n"
            for clause_name, clause_data in extracted_clauses.items():
                if clause_data.get("found"):
                    clause_context += f"- {clause_name}: {clause_data['text']}\n"
            text_for_summary += clause_context

    prompt = SUMMARY_TASK_INSTRUCTIONS.format(contract_text=text_for_summary)

    return call_text(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=MAX_TOKENS_SUMMARY,
    )