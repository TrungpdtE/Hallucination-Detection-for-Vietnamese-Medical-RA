import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import argparse
import csv
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.abspath(".cache/torch"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.eval.hallucination_input import format_hallucination_input
from src.utils.jsonl import read_jsonl


LABEL2ID = {"supported": 0, "hallucinated": 1}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_dataset(path: str, validation_ratio: float, seed: int, max_samples: int = 0) -> Dataset:
    rows = [
        {
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "context": row.get("context", ""),
            "answer": row.get("answer", ""),
            "text": format_hallucination_input(row),
            "label": LABEL2ID[row["label"]],
        }
        for row in read_jsonl(path)
        if row.get("label") in LABEL2ID
    ]
    if max_samples > 0:
        rows = rows[:max_samples]
    dataset = Dataset.from_list(rows)
    return dataset.train_test_split(test_size=validation_ratio, seed=seed)


def mean(values: List[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: List[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def print_dataset_diagnostics(dataset: Dataset, tokenizer, max_length: int) -> None:
    rows = list(dataset["train"]) + list(dataset["test"])
    labels = Counter(ID2LABEL[row["label"]] for row in rows)
    context_words = [len(row["context"].split()) for row in rows]
    answer_words = [len(row["answer"].split()) for row in rows]
    original_tokens = [token_count(tokenizer, row["text"]) + 2 for row in rows]
    truncated_tokens = [min(length, max_length) for length in original_tokens]
    cut_tokens = [max(0, length - max_length) for length in original_tokens]

    prefix_tokens = [
        token_count(
            tokenizer,
            f"Câu hỏi:\n{row['question']}\n\nCâu trả lời cần kiểm chứng:\n{row['answer']}\n\nNgữ cảnh:\n",
        )
        + 2
        for row in rows
    ]
    answer_preserved = sum(1 for length in prefix_tokens if length <= max_length)

    print("Dataset diagnostics")
    print(f"- samples: {len(rows)}")
    print(f"- labels: {dict(labels)}")
    print(f"- context words avg/p95/max: {mean(context_words):.1f}/{percentile(context_words, 0.95)}/{max(context_words, default=0)}")
    print(f"- answer words avg/p95/max: {mean(answer_words):.1f}/{percentile(answer_words, 0.95)}/{max(answer_words, default=0)}")
    print(f"- original tokens avg/p95/max: {mean(original_tokens):.1f}/{percentile(original_tokens, 0.95)}/{max(original_tokens, default=0)}")
    print(f"- after truncation avg/max: {mean(truncated_tokens):.1f}/{max(truncated_tokens, default=0)}")
    print(f"- cut tokens avg/p95/max: {mean(cut_tokens):.1f}/{percentile(cut_tokens, 0.95)}/{max(cut_tokens, default=0)}")
    print(f"- answer prefix preserved: {answer_preserved}/{len(rows)} ({answer_preserved / len(rows):.1%})")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    metrics = {"accuracy": accuracy}
    f1s = []
    for label_name, label_id in LABEL2ID.items():
        tp = int(((preds == label_id) & (labels == label_id)).sum())
        fp = int(((preds == label_id) & (labels != label_id)).sum())
        fn = int(((preds != label_id) & (labels == label_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"{label_name}_precision"] = precision
        metrics[f"{label_name}_recall"] = recall
        metrics[f"{label_name}_f1"] = f1
        f1s.append(f1)
    metrics["macro_f1"] = float(sum(f1s) / len(f1s))
    return metrics


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def balanced_class_weights(labels: List[int]) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (len(LABEL2ID) * max(1, counts[label_id])) for label_id in range(len(LABEL2ID))]
    return torch.tensor(weights, dtype=torch.float)


def write_error_analysis(path: str, rows: List[Dict], predictions) -> None:
    logits = predictions.predictions
    labels = predictions.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "context", "answer", "ground_truth", "prediction", "confidence"],
        )
        writer.writeheader()
        for row, label_id, pred_id, prob in zip(rows, labels, preds, probs):
            if int(label_id) == int(pred_id):
                continue
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "context": row.get("context", ""),
                    "answer": row.get("answer", ""),
                    "ground_truth": ID2LABEL[int(label_id)],
                    "prediction": ID2LABEL[int(pred_id)],
                    "confidence": f"{float(prob[int(pred_id)]):.4f}",
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/benchmarks/hallucination_vi.jsonl")
    parser.add_argument("--model", default="vinai/phobert-base-v2")
    parser.add_argument("--output_dir", default="outputs/models/hallucination_detector")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=0, help="Use a small subset for smoke tests; 0 means full dataset")
    parser.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer from local Hugging Face cache only")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--class_weights", action="store_true", help="Use balanced class weights in the loss")
    parser.add_argument("--error_analysis", default="", help="CSV path for validation errors")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.train, args.validation_ratio, args.seed, args.max_samples)
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_max_positions = getattr(config, "max_position_embeddings", None)
    effective_max_length = args.max_length
    if model_max_positions:
        effective_max_length = min(args.max_length, max(8, model_max_positions - 2))
    print(f"Using max_length={effective_max_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    print_dataset_diagnostics(dataset, tokenizer, effective_max_length)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=effective_max_length)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["id", "question", "context", "answer", "text"],
    )
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        local_files_only=args.local_files_only,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_strategy="epoch",
        logging_first_step=True,
        disable_tqdm=True,
    )

    class_weights = balanced_class_weights(list(tokenized["train"]["label"])) if args.class_weights else None
    if class_weights is not None:
        print(f"Using class weights: {[round(float(value), 4) for value in class_weights]}")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    print("Training...")
    trainer.train()
    print("Validation...")
    metrics = trainer.evaluate()
    predictions = trainer.predict(tokenized["test"])
    error_analysis_path = args.error_analysis or str(Path(args.output_dir) / "error_analysis.csv")
    write_error_analysis(error_analysis_path, list(dataset["test"]), predictions)
    print(f"Saved error analysis to {error_analysis_path}")
    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(metrics)
    print("Done.")


if __name__ == "__main__":
    main()
