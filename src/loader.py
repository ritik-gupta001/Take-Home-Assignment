"""
Loads contract text from CUAD, which is distributed in two possible formats:

1. Raw PDFs (full_contract_pdf/ folder) -- use load_contracts().
2. CUAD_v1.json, a SQuAD-style QA file where each contract's full text is
   already extracted into a "context" field -- use load_contracts_from_json().
   This is the more common way people download CUAD, since the JSON is the
   file most tutorials/HF dataset cards point to.

Both loaders return the same shape: {contract_id: raw_text}, so everything
downstream (preprocess, extractor, summarizer) doesn't care which one was used.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pdfplumber

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

logger = logging.getLogger(__name__)


def _extract_with_pdfplumber(pdf_path: Path) -> Optional[str]:
    try:
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
        full_text = "\n".join(pages_text).strip()
        return full_text if full_text else None
    except Exception as e:
        logger.warning(f"pdfplumber failed on {pdf_path.name}: {e}")
        return None


def _extract_with_pymupdf(pdf_path: Path) -> Optional[str]:
    if not _HAS_PYMUPDF:
        return None
    try:
        doc = fitz.open(pdf_path)
        pages_text = [page.get_text() for page in doc]
        doc.close()
        full_text = "\n".join(pages_text).strip()
        return full_text if full_text else None
    except Exception as e:
        logger.warning(f"PyMuPDF failed on {pdf_path.name}: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract full text from a single contract PDF.
    Tries pdfplumber first (better layout fidelity), falls back to PyMuPDF.
    Raises ValueError if both extractors fail or return empty text
    (this usually means the PDF is a scanned image and needs OCR, which
    is out of scope for this assignment's CUAD subset).
    """
    text = _extract_with_pdfplumber(pdf_path)
    if text is None:
        text = _extract_with_pymupdf(pdf_path)
    if text is None:
        raise ValueError(
            f"Could not extract text from {pdf_path.name}. "
            "It may be a scanned/image-only PDF requiring OCR."
        )
    return text


def load_contracts(contract_dir: Path) -> dict:
    """
    Load and extract text for every PDF in a directory.
    Returns {contract_id: raw_text}. Skips files that fail extraction
    (logged as warnings) so one bad PDF doesn't kill the whole batch.
    """
    contracts = {}
    pdf_files = sorted(contract_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {contract_dir}")

    for pdf_path in pdf_files:
        contract_id = pdf_path.stem
        try:
            contracts[contract_id] = extract_text_from_pdf(pdf_path)
        except ValueError as e:
            logger.warning(f"Skipping {contract_id}: {e}")

    logger.info(f"Loaded {len(contracts)}/{len(pdf_files)} contracts from {contract_dir}")
    return contracts


def load_contracts_from_json(json_path: Path, limit: int = None) -> dict:
    """
    Load contract text directly from CUAD_v1.json.

    CUAD_v1.json follows the SQuAD 2.0 format:
        {
          "data": [
            {
              "title": "SomeContractName",
              "paragraphs": [
                {
                  "context": "<full contract text>",
                  "qas": [ ... clause questions/answers, not needed here ... ]
                }
              ]
            },
            ...
          ]
        }

    Each "title" entry is one contract. In practice each contract has a single
    paragraph containing the full context, but we defensively concatenate all
    paragraphs' context under one title just in case a given release splits it.

    Returns {contract_id: raw_text}, same shape as load_contracts().
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    contracts = {}
    entries = cuad_data.get("data", [])

    if limit:
        entries = entries[:limit]

    for entry in entries:
        contract_id = entry.get("title", "unknown_contract")
        paragraphs = entry.get("paragraphs", [])
        # Concatenate context across paragraphs (usually just one) and
        # de-duplicate in case the same context repeats per QA group.
        seen_contexts = []
        for para in paragraphs:
            context = para.get("context", "")
            if context and context not in seen_contexts:
                seen_contexts.append(context)
        contracts[contract_id] = "\n\n".join(seen_contexts)

    logger.info(f"Loaded {len(contracts)} contracts from {json_path}")
    return contracts


def load_contracts_auto(input_path: Path, limit: int = None) -> dict:
    """
    Convenience wrapper: auto-detects whether input_path is a CUAD JSON file
    or a directory of PDFs, and calls the right loader.
    """
    input_path = Path(input_path)

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        return load_contracts_from_json(input_path, limit=limit)

    if input_path.is_dir():
        contracts = load_contracts(input_path)
        if limit:
            contracts = dict(list(contracts.items())[:limit])
        return contracts

    raise ValueError(
        f"Could not determine input type for {input_path}. "
        "Expected a .json file (CUAD_v1.json) or a directory of PDFs."
    )
