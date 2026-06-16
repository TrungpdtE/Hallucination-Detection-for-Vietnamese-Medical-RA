# Vietnamese Medical RAG Hallucination Report

## Summary

This report evaluates hallucination detection on the Vietnamese medical RAG benchmark. The task is binary classification:

`question + context + answer -> supported | hallucinated`


PhoBERT results are not available yet. Train and predict with `src/finetune/train_hallucination_detector.py` and `src/eval/predict_hallucination_transformer.py` to create `outputs/predictions/phobert_hallucination_detector.jsonl`.

## Metrics

| System | Accuracy | Macro-F1 | Supported F1 | Hallucinated F1 | N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5323 | 0.4576 | 0.6589 | 0.2563 | 6588 |

## Charts

![metric_comparison](metric_comparison.png)
![confusion_baseline](confusion_baseline.png)
![hallucination_types](hallucination_types.png)

## Interpretation

- The baseline uses lexical overlap, number consistency, and bag-of-words similarity. It is intentionally cheap and explains obvious failures.
- PhoBERT is the main Transformer judge. It should improve recall for hallucinated answers because it can model the full question-context-answer relation rather than only token overlap.
- The benchmark includes controlled hallucinations such as negation flips, number shifts, phrase drops, and unsupported appended claims.

## Next Experiment

Run PhoBERT training on GPU, then regenerate this report:

```bash
python -m src.finetune.train_hallucination_detector \
  --train data/benchmarks/hallucination_vi.jsonl \
  --model vinai/phobert-base-v2 \
  --output_dir outputs/models/phobert_hallucination_detector

python -m src.eval.predict_hallucination_transformer \
  --input data/benchmarks/hallucination_vi.jsonl \
  --model outputs/models/phobert_hallucination_detector \
  --output outputs/predictions/phobert_hallucination_detector.jsonl

python -m src.eval.generate_report
```
