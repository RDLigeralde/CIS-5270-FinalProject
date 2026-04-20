import sys
import os
import threading
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STUDENT_MODEL, TEACHER_MODEL,
    CORRECTNESS_WEIGHT, STYLE_WEIGHT,
    DATA_DIR,
)
from scripts.utils import get_openai_client, load_jsonl, save_jsonl
from llm_judge import judge_style
from verified_rewards import static_style_score, HumanDistribution

from evalplus.data import get_human_eval_plus
from tqdm import tqdm

RESULTS_FILE = f"{DATA_DIR}/benchmark_results.jsonl"
HUMAN_DIST_FILE = f"{DATA_DIR}/human_dist.json"

SYSTEM_PROMPT = (
    "You are an expert Python programmer who refactors verbose, LLM-generated code "
    "into clean, idiomatic Python. Preserve all functionality exactly."
)


def _load_model_ids() -> dict[str, str]:
    ids = {"base": STUDENT_MODEL}
    for tag, fname in [("sft", "sft_model_id.txt"), ("dpo", "dpo_model_id.txt"), ("rft", "rft_model_id.txt")]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                mid = f.read().strip()
            if mid:
                ids[tag] = mid
    return ids


def _safe_exec(code: str, test_code: str, timeout: int = 10) -> bool:
    ns: dict = {}
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


def _strip_fences(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def generate_refactored(
    client,
    deployment: str,
    problem_desc: str,
    llm_code: str,
) -> Optional[str]:
    try:
        user_msg = (
            f"Refactor the following LLM-written solution to be more idiomatic and human-like.\n\n"
            f"Problem:\n{problem_desc.strip()}\n\n"
            f"LLM solution:\n```python\n{llm_code.strip()}\n```"
        )
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return _strip_fences(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"    Inference error ({deployment}): {e}")
        return None


def evaluate_model(
    client,
    deployment: str,
    test_problems: list[dict],
    human_dist: HumanDistribution,
    label: str,
) -> dict:
    """Run full evaluation for one model deployment. Returns aggregate metrics."""
    correctness_scores, llm_style_scores, static_style_scores, rewards = [], [], [], []

    for p in tqdm(test_problems, desc=f"  {label}", leave=False):
        problem_desc = p["prompt"]
        canonical = p["canonical_solution"]
        test_code = p["test_code"]

        # Use the first verbose LLM solution input to refactor
        verbose_sols = [s for s in p.get("llm_solutions", []) if s.get("style") == "verbose" and s.get("is_correct")]
        if not verbose_sols:
            continue
        input_code = verbose_sols[0]["code"]

        generated = generate_refactored(client, deployment, problem_desc, input_code)
        if generated is None:
            continue

        correctness = 1.0 if _safe_exec(generated, test_code) else 0.0
        llm_style = judge_style(client, problem_desc, canonical, generated)
        static_style = static_style_score(generated, human_dist)
        reward = CORRECTNESS_WEIGHT * correctness + STYLE_WEIGHT * llm_style

        correctness_scores.append(correctness)
        llm_style_scores.append(llm_style)
        static_style_scores.append(static_style)
        rewards.append(reward)

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "model": label,
        "deployment": deployment,
        "n_evaluated": len(correctness_scores),
        "correctness": mean(correctness_scores),
        "llm_style": mean(llm_style_scores),
        "static_style": mean(static_style_scores),
        "reward": mean(rewards),
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Model':<10} {'Correct':>8} {'LLM Style':>10} {'StaticStyle':>12} {'Reward':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<10} "
            f"{r['correctness']:>8.3f} "
            f"{r['llm_style']:>10.3f} "
            f"{r['static_style']:>12.3f} "
            f"{r['reward']:>8.3f}"
        )
    print("=" * len(header))


def main(n_problems: int = 50):
    client = get_openai_client()
    model_ids = _load_model_ids()
    print(f"Models to evaluate: {list(model_ids.keys())}")

    # Load raw test solutions
    raw_solutions_path = f"{DATA_DIR}/llm_solutions.jsonl"
    if not os.path.exists(raw_solutions_path):
        raise FileNotFoundError(f"Run generate_solutions.py first: {raw_solutions_path}")
    all_problems = load_jsonl(raw_solutions_path)

    evalplus_problems = get_human_eval_plus()
    for p in all_problems:
        if "test_code" not in p:
            ep = evalplus_problems.get(p["task_id"], {})
            p["test_code"] = ep.get("test", "")

    test_problems = all_problems[-n_problems:]
    print(f"Evaluating on {len(test_problems)} test problems...")

    # Load or build human distribution for JSD style
    if os.path.exists(HUMAN_DIST_FILE):
        human_dist = HumanDistribution.load(HUMAN_DIST_FILE)
    else:
        print("Building human distribution from canonical solutions...")
        train_problems = all_problems[:-n_problems]
        canonicals = [p["canonical_solution"] for p in train_problems]
        human_dist = HumanDistribution.from_solutions(canonicals)
        human_dist.save(HUMAN_DIST_FILE)

    results = []
    for label, deployment in model_ids.items():
        print(f"\nEvaluating {label} ({deployment})...")
        metrics = evaluate_model(client, deployment, test_problems, human_dist, label)
        results.append(metrics)
        print(
            f"  correctness={metrics['correctness']:.3f}  "
            f"llm_style={metrics['llm_style']:.3f}  "
            f"reward={metrics['reward']:.3f}"
        )

    print_table(results)
    save_jsonl(results, RESULTS_FILE)
    print(f"\nFull results saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of test problems")
    args = parser.parse_args()
    main(n_problems=args.n)
