import sys
import os
import threading
import argparse
import re
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STUDENT_MODEL, TEACHER_MODEL,
    CORRECTNESS_WEIGHT, STYLE_WEIGHT,
    DATA_DIR,
)
from scripts.utils import get_openai_client, load_jsonl, save_jsonl
from scripts.prepare_dataset import _to_executable_target
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


def _exec_with_error(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str | None]:
    ns: dict = {}
    passed = [False]
    error_msg = [None]
    # add common typing names for annotated model outputs
    runtime_code = "from typing import *\n\n" + code

    def run():
        try:
            exec(compile(runtime_code, "<code>", "exec"), ns)
            exec(compile(test_code, "<test>", "exec"), ns)
            passed[0] = True
        except Exception as exc:
            error_msg[0] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, f"TimeoutError: execution exceeded {timeout}s"
    return passed[0], error_msg[0]


def _strip_fences(code: str) -> str:
    """Best-effort extraction of Python source from chatty model output."""
    text = code.strip()

    # Prefer fenced code blocks when present.
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()

    # If the model added explanations before code, cut to first likely code line.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith(("from ", "import ", "def ", "class ", "@", "if __name__")):
            return "\n".join(lines[i:]).strip()

    # Fallback: return raw text so downstream logging still captures the output.
    return text


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
    case_records: list[dict] | None = None,
    only_failures: bool = False,
) -> dict:
    """Run full evaluation for one model deployment. Returns aggregate metrics."""
    correctness_scores, llm_style_scores, static_style_scores, rewards = [], [], [], []

    for p in tqdm(test_problems, desc=f"  {label}", leave=False):
        problem_desc = p["prompt"]
        canonical = _to_executable_target(problem_desc, p["canonical_solution"])
        test_code = p["test_code"]

        # Use the first verbose LLM solution input to refactor
        verbose_sols = [s for s in p.get("llm_solutions", []) if s.get("style") == "verbose" and s.get("is_correct")]
        if not verbose_sols:
            continue
        input_code = verbose_sols[0]["code"]

        generated = generate_refactored(client, deployment, problem_desc, input_code)
        if generated is None:
            continue

        passed, exec_error = _exec_with_error(generated, test_code)
        correctness = 1.0 if passed else 0.0
        llm_style = judge_style(client, problem_desc, canonical, generated)
        static_style = static_style_score(generated, human_dist)
        reward = CORRECTNESS_WEIGHT * correctness + STYLE_WEIGHT * llm_style

        if case_records is not None and (not only_failures or not passed):
            case_records.append({
                "model": label,
                "deployment": deployment,
                "task_id": p.get("task_id", ""),
                "passed": passed,
                "exec_error": exec_error,
                "correctness": correctness,
                "llm_style": llm_style,
                "static_style": static_style,
                "reward": reward,
                "prompt": problem_desc,
                "input_code": input_code,
                "generated_code": generated,
                "canonical_solution": canonical,
            })

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


def main(n_problems: int = 50, dump_cases_path: str | None = None, only_failures: bool = False):
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
    case_records: list[dict] | None = [] if dump_cases_path else None
    for label, deployment in model_ids.items():
        print(f"\nEvaluating {label} ({deployment})...")
        metrics = evaluate_model(client, deployment, test_problems, human_dist, label,
                                 case_records=case_records, only_failures=only_failures)
        results.append(metrics)
        print(
            f"  correctness={metrics['correctness']:.3f}  "
            f"llm_style={metrics['llm_style']:.3f}  "
            f"reward={metrics['reward']:.3f}"
        )

    print_table(results)
    save_jsonl(results, RESULTS_FILE)
    print(f"\nFull results saved -> {RESULTS_FILE}")
    if dump_cases_path and case_records is not None:
        save_jsonl(case_records, dump_cases_path)
        print(f"Case-level debug saved -> {dump_cases_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of test problems")
    parser.add_argument("--dump-cases", default=None, help="Optional JSONL path to dump per-case generated outputs and pass/fail details")
    parser.add_argument("--only-failures", action="store_true", default=False, help="When used with --dump-cases, save only failed cases")
    args = parser.parse_args()
    main(n_problems=args.n, dump_cases_path=args.dump_cases, only_failures=args.only_failures)
