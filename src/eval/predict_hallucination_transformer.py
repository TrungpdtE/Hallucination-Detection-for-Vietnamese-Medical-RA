import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.abspath(".cache/torch"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.eval.hallucination_input import format_hallucination_input
from src.eval.metrics import LABELS, compute_classification_metrics, confusion_matrix
from src.utils.jsonl import read_jsonl, write_jsonl


ID2LABEL = {0: "supported", 1: "hallucinated"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


def batched(rows: List[Dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def print_classification_report(rows: List[Dict]) -> None:
    metrics = compute_classification_metrics(rows)
    matrix = confusion_matrix(rows)
    print("Classification report")
    print("label precision recall f1")
    for label in LABELS:
        print(
            f"{label} "
            f"{metrics[f'{label}_precision']:.4f} "
            f"{metrics[f'{label}_recall']:.4f} "
            f"{metrics[f'{label}_f1']:.4f}"
        )
    print(f"accuracy {metrics['accuracy']:.4f}")
    print(f"macro_f1 {metrics['macro_f1']:.4f}")
    print("Confusion matrix")
    print(f"gold\\pred {LABELS[0]} {LABELS[1]}")
    for gold in LABELS:
        print(f"{gold} {matrix[gold][LABELS[0]]} {matrix[gold][LABELS[1]]}")


def write_error_analysis(path: str, rows: List[Dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "context", "answer", "ground_truth", "prediction", "confidence"],
        )
        writer.writeheader()
        for row in rows:
            if row.get("label") == row.get("pred_label"):
                continue
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "context": row.get("context", ""),
                    "answer": row.get("answer", ""),
                    "ground_truth": row.get("label", ""),
                    "prediction": row.get("pred_label", ""),
                    "confidence": row.get("confidence", ""),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/benchmarks/hallucination_vi.jsonl")
    parser.add_argument("--model", default="outputs/models/phobert_hallucination_detector")
    parser.add_argument("--output", default="outputs/predictions/phobert_hallucination_detector.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0, help="Predict a subset for smoke tests; 0 means full dataset")
    parser.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer from local cache only")
    parser.add_argument("--error_analysis", default="", help="CSV path for prediction errors")
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
            prob_rows = probs.cpu().tolist()
            for row, pred_id, conf, prob in zip(batch, pred_ids, confidence, prob_rows):
                predictions.append(
                    {
                        **row,
                        "pred_label": ID2LABEL.get(pred_id, str(pred_id)),
                        "confidence": round(float(conf), 4),
                        "supported_prob": round(float(prob[LABEL2ID["supported"]]), 4),
                        "hallucinated_prob": round(float(prob[LABEL2ID["hallucinated"]]), 4),
                        "model_name": args.model,
                    }
                )

    write_jsonl(args.output, predictions)
    error_analysis_path = args.error_analysis or str(Path(args.output).with_suffix(".errors.csv"))
    write_error_analysis(error_analysis_path, predictions)
    print_classification_report(predictions)
    print(f"Saved predictions to {args.output}")
    print(f"Saved error analysis to {error_analysis_path}")


if __name__ == "__main__":
    main()
