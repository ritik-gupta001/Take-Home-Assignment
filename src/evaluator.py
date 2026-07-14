"""
Scores LLM clause extraction against CUAD's human-annotated ground truth,
giving an objective accuracy signal instead of relying purely on manual
spot-checks.

Ground truth is read from data/subset_50.json, which already has
termination_clause / liability_clause ground truth embedded per contract
(pulled directly from CUAD_v1.json's QA annotations when the subset was
built). This is the file this pipeline actually ships with -- CUAD's
separate master_clauses.csv is not required.

Uses token-overlap F1 (similar in spirit to SQuAD's metric) since exact
string match is too strict for clause spans extracted by two different
processes (human annotation vs. LLM quoting).
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import DATA_DIR

SUBSET_GROUND_TRUTH_JSON = DATA_DIR / "subset_50.json"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(predicted: str, ground_truth: str) -> float:
    """Token-level F1 overlap between predicted and ground-truth clause text."""
    pred_tokens = _tokenize(predicted)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(gt_tokens))
    return 2 * precision * recall / (precision + recall)


def load_ground_truth(json_path: Path = SUBSET_GROUND_TRUTH_JSON) -> Dict[str, dict]:
    """
    Load ground truth from data/subset_50.json.
    Returns {contract_id: {"termination_clause": str|None, "liability_clause": str|None}}.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        subset = json.load(f)
    return {entry["contract_id"]: entry["ground_truth"] for entry in subset}


def evaluate_contract(
    contract_id: str,
    predicted_clauses: Dict[str, dict],
    ground_truth_lookup: Dict[str, dict],
) -> Dict[str, float]:
    """
    Compare one contract's predicted clauses against CUAD ground truth.
    Returns {clause_type: f1_score}. Score is None if the contract isn't in
    the ground-truth set, or if both prediction and ground truth agree the
    clause is absent (trivially correct, not a meaningful F1 signal).

    NOTE: no ground truth exists for confidentiality_clause -- CUAD's 41
    official categories don't include a standalone Confidentiality type.
    """
    gt = ground_truth_lookup.get(contract_id)
    if gt is None:
        return {"termination_clause": None, "liability_clause": None}

    scores = {}
    for clause_key in ["termination_clause", "liability_clause"]:
        gt_text = gt.get(clause_key) or ""
        predicted_text = (predicted_clauses.get(clause_key) or {}).get("text", "") or ""
        if not gt_text and not predicted_text:
            scores[clause_key] = None  # both agree "not present" -- not a useful F1 signal
        else:
            scores[clause_key] = token_f1(predicted_text, gt_text)

    return scores


def evaluate_all(results: List[dict], ground_truth_json: Path = SUBSET_GROUND_TRUTH_JSON) -> pd.DataFrame:
    """
    results: list of dicts with keys {contract_id, termination_clause, ...}
             (as produced by run_pipeline.py)
    Returns a DataFrame with per-contract F1 scores for termination_clause
    and liability_clause.
    """
    ground_truth_lookup = load_ground_truth(ground_truth_json)
    rows = []

    for result in results:
        scores = evaluate_contract(result["contract_id"], result, ground_truth_lookup)
        scores["contract_id"] = result["contract_id"]
        rows.append(scores)

    eval_df = pd.DataFrame(rows)
    return eval_df