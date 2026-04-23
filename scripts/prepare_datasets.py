"""
SFT format:
  { "messages": [system, user(llm_solution), assistant(human_ref)] }

DPO format:
  { "input": [system, user], "preferred_output": [...], "non_preferred_output": [...] }

  Two pair types per the proposal:
    Positive: chosen=human_ref,     rejected=correct_llm   (teaches style)
    Negative: chosen=correct_llm,   rejected=incorrect_llm (teaches correctness)

RFT format:
  { "input": [system, user], "reference_solution": "...", "test_code": "...", "problem_description": "..." }
"""

import sys
import os
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_SOLUTIONS_FILE,
    SFT_TRAIN_FILE, SFT_VAL_FILE,
    DPO_TRAIN_FILE, DPO_VAL_FILE,
    RFT_TRAIN_FILE, RFT_VAL_FILE,
    VAL_SPLIT,
)
from scripts.utils import load_jsonl, save_jsonl

SYSTEM_PROMPT = (
    "You are an expert Python programmer who refactors verbose, LLM-generated code "
    "into clean, idiomatic Python. Preserve all functionality exactly while making "
    "the code more concise and human-like."
)


def _user_message(problem_description: str, llm_code: str) -> str:
    return (
        f"Refactor the following LLM-written solution to be more idiomatic and human-like.\n"
        f"The solution must remain functionally correct.\n\n"
        f"Problem:\n{problem_description.strip()}\n\n"
        f"LLM solution to refactor:\n```python\n{llm_code.strip()}\n```"
    )


def _assistant_message(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

def make_sft_records(problems: list[dict]) -> list[dict]:
    """
    For each problem, pair every verbose (style='verbose') correct LLM solution
    with the canonical human reference as the target.
    """
    records = []
    for p in problems:
        canonical = p["canonical_solution"]
        problem_desc = p["prompt"]
        for sol in p["llm_solutions"]:
            if not sol["is_correct"] or sol["style"] != "verbose":
                continue
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_message(problem_desc, sol["code"])},
                    {"role": "assistant", "content": _assistant_message(canonical)},
                ]
            })
    return records


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

def make_dpo_records(problems: list[dict]) -> list[dict]:
    """
    Assembles DPO dataset consisting of 'prefer human style'
    and 'prefer correct solution' pairs
    """
    records = []
    for p in problems:
        canonical = p["canonical_solution"]
        problem_desc = p["prompt"]
        correct = [s for s in p["llm_solutions"] if s["is_correct"]]
        incorrect = [s for s in p["llm_solutions"] if not s["is_correct"]]

        if not correct:
            continue

        # (human_like, llm_correct) pairs
        for sol in correct:
            user_msg = _user_message(problem_desc, sol["code"])
            records.append({
                "input": {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ]
                },
                "preferred_output": [
                    {"role": "assistant", "content": _assistant_message(canonical)}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": _assistant_message(sol["code"])}
                ],
                "_pair_type": "style",
            })

        # (llm_correct, llm_incorrect) pairs
        if incorrect:
            chosen_sol = random.choice(correct)
            rejected_sol = random.choice(incorrect)
            user_msg = _user_message(problem_desc, canonical)
            records.append({
                "input": {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ]
                },
                "preferred_output": [
                    {"role": "assistant", "content": _assistant_message(chosen_sol["code"])}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": _assistant_message(rejected_sol["code"])}
                ],
                "_pair_type": "correctness",
            })

    return records


# ---------------------------------------------------------------------------
# RFT
# ---------------------------------------------------------------------------

def make_rft_records(problems: list[dict]) -> list[dict]:
    """
    Directly pass a verbose LLM solution
    to be corrected by PPO policy
    """
    records = []
    for p in problems:
        canonical = p["canonical_solution"]
        problem_desc = p["prompt"]
        test_code = p["test_code"]
        for sol in p["llm_solutions"]:
            if not sol["is_correct"] or sol["style"] != "verbose":
                continue
            records.append({
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_message(problem_desc, sol["code"])},
                ],
                "reference_solution": canonical,
                "test_code": test_code,
                "problem_description": problem_desc,
            })
    return records


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_and_save(
    records: list[dict],
    train_path: str,
    val_path: str,
    val_fraction: float = VAL_SPLIT,
    seed: int = 42,
) -> None:
    random.seed(seed)
    random.shuffle(records)
    n_val = max(1, int(len(records) * val_fraction))
    save_jsonl(records[n_val:], train_path)
    save_jsonl(records[:n_val], val_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT / DPO / RFT datasets.")
    parser.add_argument("--input", default=RAW_SOLUTIONS_FILE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading raw solutions from {args.input}...")
    problems = load_jsonl(args.input)
    print(f"  {len(problems)} problems loaded.")

    # SFT
    sft = make_sft_records(problems)
    print(f"SFT records: {len(sft)}")
    split_and_save(sft, SFT_TRAIN_FILE, SFT_VAL_FILE)

    # DPO
    dpo = make_dpo_records(problems)
    style_pairs = sum(1 for r in dpo if r.get("_pair_type") == "style")
    correct_pairs = sum(1 for r in dpo if r.get("_pair_type") == "correctness")
    print(f"DPO records: {len(dpo)} ({style_pairs} style, {correct_pairs} correctness)")
    for r in dpo:
        r.pop("_pair_type", None)
    split_and_save(dpo, DPO_TRAIN_FILE, DPO_VAL_FILE)

    # RFT
    rft = make_rft_records(problems)
    print(f"RFT records: {len(rft)}")
    split_and_save(rft, RFT_TRAIN_FILE, RFT_VAL_FILE)


if __name__ == "__main__":
    main()
