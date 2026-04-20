import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEACHER_MODEL

SYSTEM_PROMPT = """\
You are an expert Python code reviewer specializing in code style and human-likeness.
Your task is to evaluate how "human-like" a refactored piece of code is compared to
a human-written reference solution for the same problem.

Human-like code is typically:
  - Concise and avoids unnecessary boilerplate or defensive checks
  - Uses idiomatic Python (list comprehensions, unpacking, built-ins)
  - Has short, meaningful variable names (not overly verbose)
  - Does not over-comment or repeat itself
  - Reflects natural problem-solving intuition rather than mechanical safety

Respond with a single integer score from 1 to 5 followed by a one-sentence justification.
Format: <score>\n<justification>
"""

USER_TEMPLATE = """\
Problem description:
{problem}

Human reference solution:
```python
{reference}
```

Generated/refactored solution to evaluate:
```python
{generated}
```

How human-like is the generated solution compared to the reference? Score 1-5:
"""

LABEL_MODEL_GRADER_CONFIG = {
    "type": "label_model",
    "model": TEACHER_MODEL,
    "input": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Problem description:\n{{item.problem_description}}\n\n"
                "Human reference solution:\n```python\n{{item.reference_solution}}\n```\n\n"
                "Generated/refactored solution to evaluate:\n```python\n{{sample.output_text}}\n```\n\n"
                "How human-like is the generated solution compared to the reference? Score 1-5:"
            ),
        },
    ],
    "labels": ["1", "2", "3", "4", "5"],
    "passing_labels": ["4", "5"],
}


def judge_style(
    client,
    problem: str,
    reference: str,
    generated: str,
) -> float:
    """
    GPT-mini scores human-likeness of reference and generated solutions
    """
    prompt = USER_TEMPLATE.format(
        problem=problem.strip(),
        reference=reference.strip(),
        generated=generated.strip(),
    )
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    score = _parse_score(raw)
    return (score - 1) / 4.0 # [1, 5] -> [0.0, 1.0] normalization


def _parse_score(raw: str) -> int:
    """Extract score from judge output, default to 3 if no score found"""
    match = re.match(r"^([1-5])", raw.strip())
    if match:
        return int(match.group(1))
    digits = re.findall(r"[1-5]", raw)
    return int(digits[0]) if digits else 3


def batch_judge(client, items: list[dict]) -> list[float]:
    """
    Score a list of dicts with keys: problem, reference, generated.
    Returns a list of scores in [0, 1].
    """
    scores = []
    for i, item in enumerate(items):
        score = judge_style(
            client,
            problem=item["problem"],
            reference=item["reference"],
            generated=item["generated"],
        )
        scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"  Judged {i + 1}/{len(items)} items...")
    return scores


if __name__ == "__main__":
    import argparse
    from scripts.utils import get_openai_client, load_jsonl, save_jsonl

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with problem, reference, generated fields")
    parser.add_argument("--output", required=True, help="Output JSONL with added llm_style_score field")
    args = parser.parse_args()

    client = get_openai_client()
    records = load_jsonl(args.input)
    scores = batch_judge(client, records)

    for record, score in zip(records, scores):
        record["llm_style_score"] = score

    save_jsonl(records, args.output)
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"Average LLM style score: {avg:.3f}")
