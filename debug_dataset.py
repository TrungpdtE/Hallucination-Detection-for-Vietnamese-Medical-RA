import argparse
import os
import warnings
from collections import Counter
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoConfig, AutoTokenizer

from src.eval.hallucination_input import format_hallucination_input
from src.utils.jsonl import read_jsonl


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def percentile(values: List[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(round((len(values) - 1) * pct)))]


def print_stats(rows: List[Dict], tokenizer, max_length: int) -> None:
    labels = Counter(row.get("label", "unknown") for row in rows)
    context_tokens = [token_count(tokenizer, row.get("context", "")) for row in rows]
    answer_tokens = [token_count(tokenizer, row.get("answer", "")) for row in rows]
    original_tokens = [token_count(tokenizer, format_hallucination_input(row)) + 2 for row in rows]
    cut_tokens = [max(0, length - max_length) for length in original_tokens]
    preserved = 0
    for row in rows:
        prefix = (
            f"Câu hỏi:\n{row.get('question', '')}\n\n"
            f"Câu trả lời cần kiểm chứng:\n{row.get('answer', '')}\n\n"
            "Ngữ cảnh:\n"
        )
        if token_count(tokenizer, prefix) + 2 <= max_length:
            preserved += 1

    print(f"samples: {len(rows)}")
    print(f"labels: {dict(labels)}")
    print(f"context tokens avg/p95/max: {sum(context_tokens) / len(context_tokens):.1f}/{percentile(context_tokens, 0.95)}/{max(context_tokens)}")
    print(f"answer tokens avg/p95/max: {sum(answer_tokens) / len(answer_tokens):.1f}/{percentile(answer_tokens, 0.95)}/{max(answer_tokens)}")
    print(f"original tokens avg/p95/max: {sum(original_tokens) / len(original_tokens):.1f}/{percentile(original_tokens, 0.95)}/{max(original_tokens)}")
    print(f"cut tokens avg/p95/max: {sum(cut_tokens) / len(cut_tokens):.1f}/{percentile(cut_tokens, 0.95)}/{max(cut_tokens)}")
    print(f"answer prefix preserved: {preserved}/{len(rows)} ({preserved / len(rows):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/benchmarks/hallucination_vi.jsonl")
    parser.add_argument("--model", default="vinai/phobert-base-v2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_max_positions = getattr(config, "max_position_embeddings", None)
    max_length = min(args.max_length, max(8, model_max_positions - 2)) if model_max_positions else args.max_length
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=args.local_files_only)

    print(f"Using max_length={max_length}")
    print_stats(rows, tokenizer, max_length)

    row = rows[min(args.sample, len(rows) - 1)]
    text = format_hallucination_input(row)
    encoded = tokenizer(text, truncation=True, max_length=max_length)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    original_len = token_count(tokenizer, text) + 2
    print("\nSample")
    print(f"id: {row.get('id', '')}")
    print(f"label: {row.get('label', '')}")
    print(f"original tokens: {original_len}")
    print(f"after truncation: {len(encoded['input_ids'])}")
    print(f"cut tokens: {max(0, original_len - len(encoded['input_ids']))}")
    print("\nFormatted text")
    print(text[:2000])
    print("\nDecoded tokens")
    print(decoded[:3000])


if __name__ == "__main__":
    main()
