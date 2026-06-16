import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.eval.metrics import LABELS, compute_classification_metrics, confusion_matrix, label_counts
from src.utils.jsonl import read_jsonl


METRIC_ORDER = ["accuracy", "macro_f1", "supported_f1", "hallucinated_f1"]


def load_prediction(path: str, system_name: str) -> Optional[Tuple[str, List[Dict], Dict[str, float]]]:
    if not Path(path).exists():
        return None
    rows = read_jsonl(path)
    return system_name, rows, compute_classification_metrics(rows)


def plot_metric_comparison(systems: List[Tuple[str, List[Dict], Dict[str, float]]], output_dir: Path) -> Path:
    records = []
    for system_name, _, metrics in systems:
        for metric in METRIC_ORDER:
            records.append({"system": system_name, "metric": metric, "value": metrics.get(metric, 0.0)})
    df = pd.DataFrame(records)
    pivot = df.pivot(index="metric", columns="system", values="value").loc[METRIC_ORDER]

    ax = pivot.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0)
    ax.set_title("Hallucination Detection Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="System")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "metric_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_confusion(system_name: str, rows: List[Dict], output_dir: Path) -> Path:
    matrix = confusion_matrix(rows)
    values = [[matrix[gold][pred] for pred in LABELS] for gold in LABELS]

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Confusion Matrix - {system_name}")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = output_dir / f"confusion_{system_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_hallucination_types(rows: List[Dict], output_dir: Path) -> Path:
    hallu = [row for row in rows if row.get("label") == "hallucinated"]
    counts = label_counts(hallu, "hallucination_type")
    series = pd.Series(counts).sort_values(ascending=False)
    ax = series.plot(kind="bar", figsize=(9, 4), rot=35)
    ax.set_title("Synthetic Hallucination Types")
    ax.set_xlabel("")
    ax.set_ylabel("Examples")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "hallucination_types.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def markdown_table(systems: List[Tuple[str, List[Dict], Dict[str, float]]]) -> str:
    lines = ["| System | Accuracy | Macro-F1 | Supported F1 | Hallucinated F1 | N |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for system_name, rows, metrics in systems:
        lines.append(
            "| "
            + " | ".join(
                [
                    system_name,
                    f"{metrics.get('accuracy', 0):.4f}",
                    f"{metrics.get('macro_f1', 0):.4f}",
                    f"{metrics.get('supported_f1', 0):.4f}",
                    f"{metrics.get('hallucinated_f1', 0):.4f}",
                    str(len(rows)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    output_path: Path,
    systems: List[Tuple[str, List[Dict], Dict[str, float]]],
    chart_paths: List[Path],
    phobert_path: str,
) -> None:
    baseline = next((item for item in systems if item[0] == "Baseline"), None)
    phobert = next((item for item in systems if item[0] == "PhoBERT"), None)
    delta = ""
    if baseline and phobert:
        delta_macro = phobert[2]["macro_f1"] - baseline[2]["macro_f1"]
        delta = f"\nPhoBERT improves macro-F1 by **{delta_macro:+.4f}** over the lexical/number baseline.\n"
    elif baseline:
        delta = (
            "\nPhoBERT results are not available yet. Train and predict with "
            f"`src/finetune/train_hallucination_detector.py` and `src/eval/predict_hallucination_transformer.py` "
            f"to create `{phobert_path}`.\n"
        )

    chart_lines = "\n".join(f"![{path.stem}]({path.name})" for path in chart_paths)
    report = f"""# Vietnamese Medical RAG Hallucination Report

## Summary

This report evaluates hallucination detection on the Vietnamese medical RAG benchmark. The task is binary classification:

`question + context + answer -> supported | hallucinated`

{delta}
## Metrics

{markdown_table(systems)}

## Charts

{chart_lines}

## Interpretation

- The baseline uses lexical overlap, number consistency, and bag-of-words similarity. It is intentionally cheap and explains obvious failures.
- PhoBERT is the main Transformer judge. It should improve recall for hallucinated answers because it can model the full question-context-answer relation rather than only token overlap.
- The benchmark includes controlled hallucinations such as negation flips, number shifts, phrase drops, and unsupported appended claims.

## Next Experiment

Run PhoBERT training on GPU, then regenerate this report:

```bash
python -m src.finetune.train_hallucination_detector \\
  --train data/benchmarks/hallucination_vi.jsonl \\
  --model vinai/phobert-base-v2 \\
  --output_dir outputs/models/phobert_hallucination_detector

python -m src.eval.predict_hallucination_transformer \\
  --input data/benchmarks/hallucination_vi.jsonl \\
  --model outputs/models/phobert_hallucination_detector \\
  --output outputs/predictions/phobert_hallucination_detector.jsonl

python -m src.eval.generate_report
```
"""
    output_path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmarks/hallucination_vi.jsonl")
    parser.add_argument("--baseline", default="outputs/predictions/hallucination_baseline.jsonl")
    parser.add_argument("--phobert", default="outputs/predictions/phobert_hallucination_detector.jsonl")
    parser.add_argument("--out_dir", default="reports")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = []
    for item in [
        load_prediction(args.baseline, "Baseline"),
        load_prediction(args.phobert, "PhoBERT"),
    ]:
        if item:
            systems.append(item)

    if not systems:
        raise FileNotFoundError("No prediction files found. Run a baseline or PhoBERT prediction first.")

    chart_paths = [plot_metric_comparison(systems, output_dir)]
    chart_paths.extend(plot_confusion(system_name, rows, output_dir) for system_name, rows, _ in systems)
    if Path(args.benchmark).exists():
        chart_paths.append(plot_hallucination_types(read_jsonl(args.benchmark), output_dir))

    report_path = output_dir / "hallucination_report.md"
    write_report(report_path, systems, chart_paths, args.phobert)
    print(f"Wrote report to {report_path}")
    for path in chart_paths:
        print(f"Wrote chart to {path}")


if __name__ == "__main__":
    main()
