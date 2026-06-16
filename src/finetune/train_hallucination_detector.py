import argparse
import os
from typing import Dict

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.abspath(".cache/torch"))

import numpy as np
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
        {"text": format_hallucination_input(row), "label": LABEL2ID[row["label"]]}
        for row in read_jsonl(path)
        if row.get("label") in LABEL2ID
    ]
    if max_samples > 0:
        rows = rows[:max_samples]
    dataset = Dataset.from_list(rows)
    return dataset.train_test_split(test_size=validation_ratio, seed=seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    f1s = []
    for label_id in LABEL2ID.values():
        tp = int(((preds == label_id) & (labels == label_id)).sum())
        fp = int(((preds == label_id) & (labels != label_id)).sum())
        fn = int(((preds != label_id) & (labels == label_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
    return {"accuracy": accuracy, "macro_f1": float(sum(f1s) / len(f1s))}


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

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=effective_max_length)

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        use_safetensors=False,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
