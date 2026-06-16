import argparse
import os
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.abspath(".cache/torch"))

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.eval.hallucination_input import format_hallucination_input
from src.eval.metrics import compute_classification_metrics
from src.utils.jsonl import read_jsonl, write_jsonl


ID2LABEL = {0: "supported", 1: "hallucinated"}


def batched(rows: List[Dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/benchmarks/hallucination_vi.jsonl")
    parser.add_argument("--model", default="outputs/models/phobert_hallucination_detector")
    parser.add_argument("--output", default="outputs/predictions/phobert_hallucination_detector.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0, help="Predict a subset for smoke tests; 0 means full dataset")
    parser.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer from local cache only")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_max_positions = getattr(config, "max_position_embeddings", None)
    effective_max_length = args.max_length
    if model_max_positions:
        effective_max_length = min(args.max_length, max(8, model_max_positions - 2))
    print(f"Using max_length={effective_max_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in batched(rows, args.batch_size):
            texts = [format_hallucination_input(row) for row in batch]
            inputs = tokenizer(
                texts,
                truncation=True,
                max_length=effective_max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).cpu().tolist()
            confidence = probs.max(dim=-1).values.cpu().tolist()
            for row, pred_id, conf in zip(batch, pred_ids, confidence):
                predictions.append(
                    {
                        **row,
                        "pred_label": ID2LABEL.get(pred_id, str(pred_id)),
                        "confidence": round(float(conf), 4),
                        "model_name": args.model,
                    }
                )

    write_jsonl(args.output, predictions)
    metrics = compute_classification_metrics(predictions)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
