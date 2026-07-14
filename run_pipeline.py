"""
CLI entrypoint that runs the full document-processing pipeline:

    PDF -> extract text -> normalize -> LLM clause extraction -> LLM summary
        -> write results.csv / results.json
        -> (optional) evaluate against CUAD ground truth
        -> (optional) build semantic search index

Usage:
    python run_pipeline.py --input_dir data/subset --output_dir outputs
    python run_pipeline.py --input_dir data/subset --evaluate --build_index
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import RAW_DIR, OUTPUT_DIR, SUBSET_SIZE
from src.loader import load_contracts_auto
from src.preprocess import normalize_text
from src.extractor import extract_clauses
from src.summarizer import summarize_contract

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(input_path: Path, output_dir: Path, limit: int = None) -> list:
    """
    input_path can be either:
      - a directory of contract PDFs, or
      - a path to CUAD_v1.json (contract text already extracted into "context" fields)
    load_contracts_auto() detects which one it is.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading contracts from {input_path} ...")
    contracts = load_contracts_auto(input_path, limit=limit)

    logger.info(f"Processing {len(contracts)} contracts.")
    results = []

    for contract_id, raw_text in tqdm(contracts.items(), desc="Processing contracts"):
        try:
            clean_text = normalize_text(raw_text)

            clauses = extract_clauses(clean_text)
            summary = summarize_contract(clean_text, extracted_clauses=clauses)

            results.append({
                "contract_id": contract_id,
                "summary": summary,
                "termination_clause": clauses["termination_clause"],
                "confidentiality_clause": clauses["confidentiality_clause"],
                "liability_clause": clauses["liability_clause"],
            })
        except Exception as e:
            logger.error(f"Failed to process {contract_id}: {e}")
            results.append({
                "contract_id": contract_id,
                "summary": None,
                "termination_clause": None,
                "confidentiality_clause": None,
                "liability_clause": None,
                "error": str(e),
            })

    return results


def save_results(results: list, output_dir: Path):
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {json_path}")

    flat_rows = []
    for r in results:
        flat_rows.append({
            "contract_id": r["contract_id"],
            "summary": r.get("summary"),
            "termination_clause": (r.get("termination_clause") or {}).get("text", ""),
            "confidentiality_clause": (r.get("confidentiality_clause") or {}).get("text", ""),
            "liability_clause": (r.get("liability_clause") or {}).get("text", ""),
        })
    csv_path = output_dir / "results.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="CUAD LLM contract processing pipeline")
    parser.add_argument(
        "--input_dir", type=Path, default=RAW_DIR,
        help="Directory of contract PDFs, OR a path to CUAD_v1.json"
    )
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Directory to write results")
    parser.add_argument("--limit", type=int, default=SUBSET_SIZE, help="Max number of contracts to process")
    parser.add_argument("--evaluate", action="store_true", help="Score extraction against CUAD ground truth")
    parser.add_argument("--build_index", action="store_true", help="Build semantic search index over clauses (bonus)")
    args = parser.parse_args()

    results = run_pipeline(args.input_dir, args.output_dir, limit=args.limit)
    save_results(results, args.output_dir)

    if args.evaluate:
        from src.evaluator import evaluate_all
        eval_df = evaluate_all(results)
        eval_path = args.output_dir / "evaluation_scores.csv"
        eval_df.to_csv(eval_path, index=False)
        logger.info(f"Wrote {eval_path}")
        logger.info(f"Mean F1 scores:\n{eval_df.drop(columns=['contract_id']).mean()}")

    if args.build_index:
        from src.embeddings import ClauseSearchIndex
        index = ClauseSearchIndex()
        index.index_clauses(results)
        logger.info("Semantic search index built. Use ClauseSearchIndex().search(query) to query it.")


if __name__ == "__main__":
    main()
