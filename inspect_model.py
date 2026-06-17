import argparse
import os
import warnings
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.eval.hallucination_input import format_hallucination_input
from src.utils.jsonl import read_jsonl


ID2LABEL = {0: "supported", 1: "hallucinated"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/benchmarks/hallucination_vi.jsonl")
    parser.add_argument("--model", default="outputs/models/phobert_hallucination_detector")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    rows: List[Dict] = read_jsonl(args.input)[: args.num_samples]
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_max_positions = getattr(config, "max_position_embeddings", None)
    max_length = min(args.max_length, max(8, model_max_positions - 2)) if model_max_positions else args.max_length
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=args.local_files_only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Using max_length={max_length}")
    with torch.no_grad():
        for row in rows:
            inputs = tokenizer(
                format_hallucination_input(row),
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            ).to(device)
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu()
            pred_id = int(probs.argmax().item())
            print(
                f"{row.get('id', '')}\t"
                f"gold={row.get('label', '')}\t"
                f"pred={ID2LABEL[pred_id]}\t"
                f"supported={float(probs[0]):.4f}\t"
                f"hallucinated={float(probs[1]):.4f}"
            )


if __name__ == "__main__":
    main()
