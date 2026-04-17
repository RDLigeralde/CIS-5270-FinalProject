"""
Reward composition:
  - Correctness (weight 0.7): exec solution against EvalPlus test cases
  - Style       (weight 0.3): AST-based static analysis vs human reference
    (static analysis is the training-time surrogate; LLM-as-judge used at eval)
  - TODO: tune these
"""

import sys
import os
import argparse
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STUDENT_MODEL,
    RFT_TRAIN_FILE, RFT_VAL_FILE,
    RFT_EPOCHS, BATCH_SIZE, LR_MULTIPLIER,
    CORRECTNESS_WEIGHT, STYLE_WEIGHT,
    DATA_DIR,
)
from utils import get_openai_client, upload_file, wait_for_job

DPO_MODEL_ID_PATH = f"{DATA_DIR}/dpo_model_id.txt"
RFT_MODEL_ID_PATH = f"{DATA_DIR}/rft_model_id.txt"

# ---------------------------------------------------------------------------
# Self-contained Python grader executed inside Azure's RFT evaluation sandbox.
# Receives `item` (training data row) and `sample` (model output).
# Returns a float in [0, 1].
# ---------------------------------------------------------------------------

GRADER_SOURCE = textwrap.dedent("""\
    def grade(item, sample):
        import ast
        import math
        import threading

        # ---- helpers ----

        def _strip_fences(code):
            lines = code.strip().splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\\n".join(lines)

        def _safe_exec(code, test_code, timeout=8):
            ns = {}
            passed = [False]
            def run():
                try:
                    exec(compile(code, "<code>", "exec"), ns)
                    exec(compile(test_code, "<test>", "exec"), ns)
                    passed[0] = True
                except Exception:
                    pass
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(timeout)
            return passed[0]

        def _avg_name_len(tree):
            names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
            return sum(len(n) for n in names) / max(len(names), 1)

        def _node_count(tree):
            return sum(1 for _ in ast.walk(tree))

        def _style_score(gen_code, ref_code):
            try:
                gen_tree = ast.parse(gen_code)
                ref_tree = ast.parse(ref_code)
            except SyntaxError:
                return 0.0

            gen_lines = [l for l in gen_code.splitlines() if l.strip()]
            ref_lines = [l for l in ref_code.splitlines() if l.strip()]
            line_ratio = min(len(gen_lines), len(ref_lines)) / max(len(gen_lines), len(ref_lines), 1)

            gen_nl = _avg_name_len(gen_tree)
            ref_nl = _avg_name_len(ref_tree)
            name_sim = 1.0 / (1.0 + abs(gen_nl - ref_nl))

            gen_nc = _node_count(gen_tree)
            ref_nc = _node_count(ref_tree)
            node_ratio = min(gen_nc, ref_nc) / max(gen_nc, ref_nc, 1)

            # comment density (human solutions tend to have fewer comments)
            gen_comments = sum(1 for l in gen_code.splitlines() if l.strip().startswith("#"))
            ref_comments = sum(1 for l in ref_code.splitlines() if l.strip().startswith("#"))
            comment_sim = 1.0 - abs(
                gen_comments / max(len(gen_lines), 1) - ref_comments / max(len(ref_lines), 1)
            )

            return (line_ratio + name_sim + node_ratio + comment_sim) / 4.0

        # ---- main grading ----

        code = _strip_fences(sample.output_text)
        ref = item.get("reference_solution", "")
        test = item.get("test_code", "")

        correctness = 1.0 if (test and _safe_exec(code, test)) else 0.0
        style = _style_score(code, ref) if ref else 0.5

        return 0.7 * correctness + 0.3 * style
""")


def build_grader() -> dict:
    return {
        "type": "python",
        "source": GRADER_SOURCE,
        "image_tag": "2025-05-08",
        "name": "correctness_and_style",
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _load_model_id(path: str, fallback: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            mid = f.read().strip()
        if mid:
            return mid
    print(f"Warning: {path} not found, using fallback: {fallback}")
    return fallback


def run_rft(base_model: str | None = None, wait: bool = True) -> str:
    client = get_openai_client()

    if base_model is None:
        base_model = _load_model_id(DPO_MODEL_ID_PATH, STUDENT_MODEL)

    print("=== RFT/PPO Training ===")
    print(f"Base model: {base_model}")

    train_id = upload_file(client, RFT_TRAIN_FILE)
    val_id = upload_file(client, RFT_VAL_FILE)

    grader = build_grader()
    print("Grader: Python (correctness via exec + style via AST)")

    print(f"\nCreating Reinforcement Fine-Tuning job...")
    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=train_id,
        validation_file=val_id,
        method={
            "type": "reinforcement",
            "reinforcement": {
                "grader": grader,
                "hyperparameters": {
                    "n_epochs": RFT_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate_multiplier": LR_MULTIPLIER,
                    "eval_interval": 5,
                    "eval_samples": 10,
                    "reasoning_effort": "low",
                },
            },
        },
        extra_body={"trainingType": "Standard"},
        suffix="rft-correctness-style",
    )
    print(f"Job created: {job.id}  status={job.status}")

    if not wait:
        return job.id

    completed = wait_for_job(client, job.id, poll_interval=60)
    if completed.status != "succeeded":
        raise RuntimeError(f"RFT job {job.id} ended with status: {completed.status}")

    model_id = completed.fine_tuned_model
    print(f"\nRFT complete. Fine-tuned model: {model_id}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RFT_MODEL_ID_PATH, "w") as f:
        f.write(model_id)
    print(f"Model ID saved -> {RFT_MODEL_ID_PATH}")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=None,
                        help="Base model ID (default: reads dpo_model_id.txt)")
    parser.add_argument("--wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    args = parser.parse_args()
    run_rft(base_model=args.base_model, wait=args.wait)
