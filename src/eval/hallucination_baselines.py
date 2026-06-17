import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import argparse
import math
import re
from collections import Counter
from typing import Dict, List

from src.eval.metrics import compute_classification_metrics
from src.utils.jsonl import read_jsonl, write_jsonl


def tokenize(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


def lexical_support(context: str, answer: str) -> float:
    answer_terms = [t for t in tokenize(answer) if len(t) > 2]
    if not answer_terms:
        return 0.0
    context_terms = set(tokenize(context))
    return sum(1 for term in answer_terms if term in context_terms) / len(answer_terms)


def number_consistency(context: str, answer: str) -> float:
    answer_nums = set(re.findall(r"\d+(?:[.,]\d+)?", answer))
    if not answer_nums:
        return 1.0
    context_nums = set(re.findall(r"\d+(?:[.,]\d+)?", context))
    return sum(1 for n in answer_nums if n in context_nums) / len(answer_nums)


def cosine_bow(context: str, answer: str) -> float:
    c1 = Counter(tokenize(context))
    c2 = Counter(tokenize(answer))
    common = set(c1) & set(c2)
    dot = sum(c1[t] * c2[t] for t in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def predict(row: Dict, threshold: float) -> Dict:
    context = row.get("context", "")
    answer = row.get("answer", "")
    lex = lexical_support(context, answer)
    nums = number_consistency(context, answer)
    bow = cosine_bow(context, answer)
    score = 0.55 * lex + 0.30 * nums + 0.15 * bow
    pred = "supported" if score >= threshold else "hallucinated"
    return {
        **row,
        "pred_label": pred,
        "support_score": round(score, 4),
        "features": {
            "lexical_support": round(lex, 4),
            "number_consistency": round(nums, 4),
            "bow_cosine": round(bow, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions/hallucination_baseline.jsonl")
    parser.add_argument("--threshold", type=float, default=0.68)
    args = parser.parse_args()

    rows = [predict(row, args.threshold) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    metrics = compute_classification_metrics(rows)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
