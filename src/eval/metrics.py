from collections import Counter
from typing import Dict, List


LABELS = ["supported", "hallucinated"]


def compute_classification_metrics(rows: List[Dict], label_key: str = "label", pred_key: str = "pred_label") -> Dict[str, float]:
    metrics = {}
    total = len(rows)
    correct = sum(1 for row in rows if row.get(label_key) == row.get(pred_key))
    metrics["accuracy"] = correct / total if total else 0.0

    macro_f1 = 0.0
    for label in LABELS:
        tp = sum(1 for row in rows if row.get(label_key) == label and row.get(pred_key) == label)
        fp = sum(1 for row in rows if row.get(label_key) != label and row.get(pred_key) == label)
        fn = sum(1 for row in rows if row.get(label_key) == label and row.get(pred_key) != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall
        metrics[f"{label}_f1"] = f1
        macro_f1 += f1

    metrics["macro_f1"] = macro_f1 / len(LABELS)
    return metrics


def confusion_matrix(rows: List[Dict], label_key: str = "label", pred_key: str = "pred_label") -> Dict[str, Dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        gold = row.get(label_key)
        pred = row.get(pred_key)
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def label_counts(rows: List[Dict], key: str = "label") -> Dict[str, int]:
    counts = Counter(row.get(key, "unknown") for row in rows)
    return dict(sorted(counts.items()))
