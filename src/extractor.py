"""
Uses the configured LLM provider (Anthropic or OpenAI, see config.LLM_PROVIDER)
with forced structured output to extract termination, confidentiality, and
liability clauses from contract text.

For long contracts, extraction runs per-chunk and results are merged,
preferring the highest-confidence, non-empty match for each clause type.
"""

import logging
from typing import Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from config import MAX_TOKENS_EXTRACTION, CLAUSE_TYPES
from prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_TASK_INSTRUCTIONS,
    EXTRACTION_TOOL_SCHEMA,
    build_few_shot_block,
)
from src.chunker import needs_chunking, chunk_text
from src.llm_client import call_structured

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def _call_extraction_api(contract_text: str, few_shot_examples: List[dict] = None) -> dict:
    """Single API call: extract clauses from one (chunk of) contract text."""
    few_shot_block = build_few_shot_block(few_shot_examples or [])
    prompt = few_shot_block + EXTRACTION_TASK_INSTRUCTIONS.format(contract_text=contract_text)

    return call_structured(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=prompt,
        tool_schema=EXTRACTION_TOOL_SCHEMA,
        max_tokens=MAX_TOKENS_EXTRACTION,
    )


def _merge_chunk_results(chunk_results: List[dict]) -> dict:
    """
    Merge per-chunk extraction results, keeping the highest-confidence
    non-empty match for each clause type across all chunks.
    """
    merged = {
        clause: {"found": False, "text": "", "confidence": "low"}
        for clause in CLAUSE_TYPES
    }

    for result in chunk_results:
        for clause in CLAUSE_TYPES:
            candidate = result.get(clause, {})
            if not candidate.get("found"):
                continue
            current_rank = _CONFIDENCE_RANK.get(merged[clause]["confidence"], 0)
            candidate_rank = _CONFIDENCE_RANK.get(candidate.get("confidence", "low"), 0)
            if not merged[clause]["found"] or candidate_rank > current_rank:
                merged[clause] = candidate

    return merged


def extract_clauses(contract_text: str, few_shot_examples: List[dict] = None) -> Dict:
    """
    Extract termination, confidentiality, and liability clauses from a contract.
    Handles chunking transparently for long contracts.
    """
    if needs_chunking(contract_text):
        logger.info("Contract exceeds context budget, splitting into chunks for extraction.")
        chunks = chunk_text(contract_text)
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  extracting from chunk {i + 1}/{len(chunks)}")
            chunk_results.append(_call_extraction_api(chunk, few_shot_examples))
        return _merge_chunk_results(chunk_results)

    return _call_extraction_api(contract_text, few_shot_examples)