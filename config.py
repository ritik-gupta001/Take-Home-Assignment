"""
Central configuration for the CUAD LLM pipeline.
Keeping all knobs in one place makes the pipeline easy to tune and review.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # original CUAD PDFs go here
SUBSET_DIR = DATA_DIR / "subset"    # the 50-contract subset used for this run
OUTPUT_DIR = ROOT_DIR / "outputs"
CUAD_MASTER_CLAUSES_CSV = DATA_DIR / "master_clauses.csv"  # CUAD ground-truth annotations

# --- LLM -----------------------------------------------------------------
# Set LLM_PROVIDER=openai in .env if you have an OpenAI key instead of an
# Anthropic key. Defaults to whichever key is actually present in .env.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_explicit_provider = os.getenv("LLM_PROVIDER")
if _explicit_provider:
    LLM_PROVIDER = _explicit_provider.lower()
elif OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    LLM_PROVIDER = "openai"
else:
    LLM_PROVIDER = "anthropic"

MODEL_NAME = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini" if LLM_PROVIDER == "openai" else "claude-sonnet-4-6",
)
MAX_TOKENS_EXTRACTION = 1500
MAX_TOKENS_SUMMARY = 400
TEMPERATURE = 0.0  # deterministic extraction

# --- Chunking --------------------------------------------------------
# Contracts longer than this (in tokens) get split into overlapping chunks
# before extraction, then results are merged/de-duplicated.
MAX_CONTEXT_TOKENS = 12000
CHUNK_OVERLAP_TOKENS = 500

# --- Clause types this assignment asks for ------------------------------
CLAUSE_TYPES = ["termination_clause", "confidentiality_clause", "liability_clause"]

# Map our clause keys -> CUAD's official annotation column names,
# used only by evaluator.py when scoring against ground truth.
#
# NOTE: CUAD's 41 official clause categories do NOT include a standalone
# "Confidentiality" type (verified against the actual CUAD_v1.json category
# list). The closest related official categories (Non-Disparagement, Audit
# Rights, Covenant Not To Sue) aren't true equivalents, so confidentiality_clause
# has no ground-truth column to compare against -- it's evaluated only by
# manual spot-check / LLM self-consistency, not F1 against CUAD annotations.
CUAD_COLUMN_MAP = {
    "termination_clause": "Termination For Convenience",
    "liability_clause": "Cap On Liability",
}

# --- Number of contracts to process (per assignment spec) ---------------
SUBSET_SIZE = 50