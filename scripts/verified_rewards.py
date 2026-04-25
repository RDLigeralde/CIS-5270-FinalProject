import radon.complexity as rc
import radon.metrics as rm
import radon.raw as rr
import ast
import re
import textwrap

from dataclasses import dataclass, field
from typing import Sequence
import json
import math

def _cyclomatic_complexity(code: str) -> float:
    """Average McCabe complexity (1 on parse error)"""
    try:
        blocks = rc.cc_visit(code)
        if not blocks:
            return 1.0
        return sum(b.complexity for b in blocks) / len(blocks)
    except Exception:
        return 1.0


def _halstead(code: str) -> dict:
    """Halstead volume, difficulty, and effort (0 on parse error)"""
    try:
        result = rm.h_visit(code)
        if not result:
            return {"volume": 0.0, "difficulty": 0.0, "effort": 0.0}
        total = result[0]
        return {
            "volume": total.volume,
            "difficulty": total.difficulty,
            "effort": total.effort,
        }
    except Exception:
        return {"volume": 0.0, "difficulty": 0.0, "effort": 0.0}


def _raw_metrics(code: str) -> dict:
    """Lines of code, logical lines, comment lines, blank lines"""
    try:
        m = rr.analyze(code)
        return {
            "loc": m.loc,
            "lloc": m.lloc,
            "comments": m.comments,
            "blank": m.blank,
        }
    except Exception:
        return {"loc": 0, "lloc": 0, "comments": 0, "blank": 0}


def _identifier_lengths(code: str) -> list[int]:
    """Lengths of all Name nodes in the AST (variable/function names."""
    try:
        tree = ast.parse(code)
        return [len(node.id) for node in ast.walk(tree) if isinstance(node, ast.Name)]
    except SyntaxError:
        return []


def _identifier_dist(lengths: list[int], bins: int = 20) -> list[float]:
    """Normalised histogram over identifier lengths (0..bins-1 chars)"""
    counts = [0] * bins
    for l in lengths:
        idx = min(l, bins - 1)
        counts[idx] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def _extract_function_scaffold(problem_description: str) -> tuple[list[str], str] | None:
    """Extract import lines and the first function signature from prompt text."""
    lines = problem_description.strip().splitlines()
    imports: list[str] = []
    signature: str | None = None

    for line in lines:
        s = line.strip()
        if s.startswith(("from ", "import ")):
            imports.append(s)
        if signature is None and re.match(r"^def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*(?:->\s*.*)?\s*:\s*$", s):
            signature = s

    if signature is None:
        return None
    return imports, signature


def _to_executable_target(problem_description: str, code: str) -> str:
    """Convert body-only canonical snippets into full executable function code."""
    src = code.strip()
    scaffold = _extract_function_scaffold(problem_description)
    if scaffold is None:
        return src

    imports, signature = scaffold
    if src.lstrip().startswith(signature):
        return src

    body = textwrap.dedent(code).strip("\n")
    if not body:
        body = "pass"

    indented_body = "\n".join(f"    {line}" if line.strip() else "" for line in body.splitlines())
    function_block = f"{signature}\n{indented_body}"

    if imports:
        return "\n".join(imports) + "\n\n" + function_block
    return function_block


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def _jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """JSD in [0, 1] (0 = identical distributions)."""
    eps = 1e-10
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log((pi + eps) / (mi + eps)) for pi, mi in zip(p, m))
    kl_qm = sum(qi * math.log((qi + eps) / (mi + eps)) for qi, mi in zip(q, m))
    jsd_bits = (kl_pm + kl_qm) / 2
    # Normalise so JSD ∈ [0, 1] (base-2 log gives max of 1 bit)
    return min(jsd_bits / math.log(2), 1.0)


@dataclass
class HumanDistribution:
    """Statistics derived from canonical EvalPlus human solutions"""
    avg_complexity: float = 2.0
    avg_lloc: float = 10.0
    avg_halstead_volume: float = 150.0
    avg_name_length: float = 6.0
    name_length_hist: list[float] = field(default_factory=lambda: [0.05] * 20)

    @classmethod
    def from_solutions(cls, solutions: list[str]) -> "HumanDistribution":
        complexities, llocs, volumes, all_lengths = [], [], [], []
        for code in solutions:
            complexities.append(_cyclomatic_complexity(code))
            llocs.append(_raw_metrics(code)["lloc"])
            volumes.append(_halstead(code)["volume"])
            lengths = _identifier_lengths(code)
            all_lengths.extend(lengths)

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return cls(
            avg_complexity=safe_mean(complexities),
            avg_lloc=safe_mean(llocs),
            avg_halstead_volume=safe_mean(volumes),
            avg_name_length=safe_mean(all_lengths),
            name_length_hist=_identifier_dist(all_lengths),
        )

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HumanDistribution":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Main reward computation
# ---------------------------------------------------------------------------

def static_style_score(
    generated_code: str,
    human_dist: HumanDistribution,
) -> float:
    """
    Compute a style score in [0, 1] for generated_code relative to the
    human reference distribution.  Higher = more human-like.

    Sub-scores:
      - complexity_sim : how close McCabe complexity is to human average
      - loc_sim        : how close logical lines of code is to human average
      - halstead_sim   : how close Halstead volume is to human average
      - name_len_sim   : how close avg identifier length is to human average
      - jsd_score      : 1 - JSD between identifier-length histograms
    """
    gen_complexity = _cyclomatic_complexity(generated_code)
    gen_lloc = _raw_metrics(generated_code)["lloc"]
    gen_volume = _halstead(generated_code)["volume"]
    gen_lengths = _identifier_lengths(generated_code)
    gen_name_len = sum(gen_lengths) / max(len(gen_lengths), 1)
    gen_hist = _identifier_dist(gen_lengths)

    def proximity(val, ref, scale=None):
        """Gaussian similarity: 1 when val==ref, decays away from ref."""
        if scale is None:
            scale = max(ref, 1.0)
        return math.exp(-((val - ref) ** 2) / (2 * scale ** 2))

    complexity_sim = proximity(gen_complexity, human_dist.avg_complexity, scale=max(human_dist.avg_complexity, 1))
    loc_sim = proximity(gen_lloc, human_dist.avg_lloc, scale=max(human_dist.avg_lloc, 1))
    halstead_sim = proximity(gen_volume, human_dist.avg_halstead_volume, scale=max(human_dist.avg_halstead_volume, 1))
    name_len_sim = proximity(gen_name_len, human_dist.avg_name_length, scale=max(human_dist.avg_name_length, 1))
    jsd_score = 1.0 - _jsd(gen_hist, human_dist.name_length_hist)

    score = (complexity_sim + loc_sim + halstead_sim + name_len_sim + jsd_score) / 5.0
    return float(score)


def full_static_reward(
    generated_code: str,
    is_correct: bool,
    human_dist: HumanDistribution,
    correctness_weight: float = 0.7,
    style_weight: float = 0.3,
) -> dict:
    """
    Return the full reward dict:
      {
        "correctness": 0 or 1,
        "style":       float in [0, 1],
        "reward":      weighted combination,
        "breakdown": { sub-score details }
      }
    """
    correctness = 1.0 if is_correct else 0.0
    style = static_style_score(generated_code, human_dist)
    reward = correctness_weight * correctness + style_weight * style

    return {
        "correctness": correctness,
        "style": style,
        "reward": reward,
    }


if __name__ == "__main__":
    import argparse
    try:
        from scripts.utils import load_jsonl
    except ModuleNotFoundError:
        import os
        import sys

        # Allow running as: python scripts/verified_rewards.py ...
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # If another installed package named "scripts" was imported first,
        # drop it so import resolution can use this repo's scripts package.
        sys.modules.pop("scripts", None)
        sys.modules.pop("scripts.utils", None)
        from scripts.utils import load_jsonl

    parser = argparse.ArgumentParser(description="Pre-compute human reference distribution.")
    parser.add_argument("--solutions", default="data_files/llm_solutions.jsonl",
                        help="JSONL with field 'canonical_solution'")
    parser.add_argument("--out", default="data_files/human_dist.json",
                        help="Output path for the distribution JSON")
    args = parser.parse_args()

    records = load_jsonl(args.solutions)
    canonical_wrapped = [
        _to_executable_target(r.get("prompt", ""), r["canonical_solution"])
        for r in records
        if r.get("canonical_solution")
    ]
    print(f"Computing distribution from {len(canonical_wrapped)} wrapped canonical solutions...")
    dist = HumanDistribution.from_solutions(canonical_wrapped)
    dist.save(args.out)
    print(f"Saved -> {args.out}")
    print(f"  avg_complexity : {dist.avg_complexity:.2f}")
    print(f"  avg_lloc       : {dist.avg_lloc:.2f}")
    print(f"  avg_volume     : {dist.avg_halstead_volume:.2f}")
    print(f"  avg_name_len   : {dist.avg_name_length:.2f}")
