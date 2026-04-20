import sys
import os
import threading
import time
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEACHER_MODEL, RAW_SOLUTIONS_FILE, N_SOLUTIONS_PER_PROBLEM
from scripts.utils import get_openai_client, save_jsonl

from evalplus.data import get_human_eval_plus
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VERBOSE_SYSTEM = """\
You are a cautious Python programmer who writes defensive, well-documented code.
Always include type hints, input validation, and detailed variable names.
Output only the function implementation — no explanations or test code.
"""

NATURAL_SYSTEM = """\
You are a Python programmer. Solve the given problem concisely.
Output only the function implementation — no explanations or test code.
"""

REFACTOR_SYSTEM = """\
You are an expert Python programmer. Solve the given problem as a human would:
write clean, idiomatic, Pythonic code without unnecessary boilerplate.
Output only the function implementation — no explanations or test code.
"""


def build_user_prompt(problem_prompt: str) -> str:
    return f"Implement the following Python function:\n\n{problem_prompt}"


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def _safe_exec(solution_code: str, test_code: str, timeout: int = 10) -> bool:
    """Execute solution + test code in an isolated namespace. Returns True if tests pass."""
    namespace: dict = {}
    passed = [False]

    def run():
        try:
            exec(compile(solution_code, "<solution>", "exec"), namespace)
            exec(compile(test_code, "<test>", "exec"), namespace)
            passed[0] = True
        except Exception:
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    return passed[0]


def check_solution(solution_code: str, test_code: str) -> bool:
    """Return True if solution passes EvalPlus test code."""
    return _safe_exec(solution_code, test_code)


# ---------------------------------------------------------------------------
# Solution generation
# ---------------------------------------------------------------------------

def generate_solution(
    client,
    problem_prompt: str,
    system_prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 1024,
) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(problem_prompt)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        code = response.choices[0].message.content.strip()
        return _strip_fences(code)
    except Exception as e:
        print(f"    Generation error: {e}")
        return None


def _strip_fences(code: str) -> str:
    """Remove ```python ... ``` markdown fences if present."""
    lines = code.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def generate_solutions_for_problem(
    client,
    task_id: str,
    problem_prompt: str,
    canonical_solution: str,
    test_code: str,
    n: int = N_SOLUTIONS_PER_PROBLEM,
) -> list[dict]:
    """
    Generate n solutions for one problem using a mix of styles.
    Always generates at least one verbose and one natural solution.
    """
    configs = []
    # Verbose LLM-style solutions
    n_verbose = max(1, n // 2)
    for i in range(n_verbose):
        configs.append(("verbose", VERBOSE_SYSTEM, 0.2 + i * 0.1))
    # Natural / varied solutions (some correct, some not — used for DPO negative pairs)
    n_natural = n - n_verbose
    for i in range(n_natural):
        configs.append(("natural", NATURAL_SYSTEM, 0.7 + i * 0.2))

    results = []
    for style, system, temp in configs:
        code = generate_solution(client, problem_prompt, system, temperature=min(temp, 1.5))
        if code is None:
            continue
        correct = check_solution(code, test_code)
        results.append({
            "code": code,
            "is_correct": correct,
            "temperature": temp,
            "style": style,
        })
        time.sleep(0.2) # for rate limiting

    return results


def main(n_problems: Optional[int] = None, offset: int = 0):
    client = get_openai_client()
    problems = get_human_eval_plus()

    task_ids = list(problems.keys())
    if n_problems:
        task_ids = task_ids[offset: offset + n_problems]

    print(f"Generating solutions for {len(task_ids)} EvalPlus problems using {TEACHER_MODEL}...")

    records = []
    for task_id in tqdm(task_ids, desc="Problems"):
        p = problems[task_id]
        prompt = p["prompt"]
        canonical = p["canonical_solution"]
        test_code = p["test"]
        entry_point = p["entry_point"]

        llm_solutions = generate_solutions_for_problem(
            client=client,
            task_id=task_id,
            problem_prompt=prompt,
            canonical_solution=canonical,
            test_code=test_code,
        )

        n_correct = sum(1 for s in llm_solutions if s["is_correct"])
        tqdm.write(f"  {task_id}: {n_correct}/{len(llm_solutions)} correct")

        records.append({
            "task_id": task_id,
            "prompt": prompt,
            "canonical_solution": canonical,
            "test_code": test_code,
            "entry_point": entry_point,
            "llm_solutions": llm_solutions,
        })

    save_jsonl(records, RAW_SOLUTIONS_FILE)
    total_solutions = sum(len(r["llm_solutions"]) for r in records)
    total_correct = sum(
        sum(1 for s in r["llm_solutions"] if s["is_correct"]) for r in records
    )
    print(f"\nDone. {total_solutions} solutions generated, {total_correct} correct.")
    print(f"Saved -> {RAW_SOLUTIONS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM solutions from EvalPlus.")
    parser.add_argument("--n", type=int, default=None, help="Number of problems (default: all)")
    parser.add_argument("--offset", type=int, default=0, help="Problem offset")
    args = parser.parse_args()
    main(n_problems=args.n, offset=args.offset)
