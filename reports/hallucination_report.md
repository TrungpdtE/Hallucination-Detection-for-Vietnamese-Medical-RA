# Vietnamese Medical RAG Hallucination Report

## Summary

This report evaluates hallucination detection on the Vietnamese medical RAG benchmark. The task is binary classification:

`question + answer + context -> supported | hallucinated`


PhoBERT improves macro-F1 by **+0.3684** over the lexical/number baseline.

## Metrics

| System | Accuracy | Macro-F1 | Supported F1 | Hallucinated F1 | N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5299 | 0.4347 | 0.6667 | 0.2028 | 6588 |
| PhoBERT | 0.8078 | 0.8032 | 0.8334 | 0.7730 | 6588 |

## Charts

![metric_comparison](metric_comparison.png)
![confusion_baseline](confusion_baseline.png)
![confusion_phobert](confusion_phobert.png)
![hallucination_types](hallucination_types.png)

## Interpretation

- The baseline uses lexical overlap, number consistency, and bag-of-words similarity. It is intentionally cheap and explains obvious failures.
- PhoBERT is the main Transformer judge. It should improve recall for hallucinated answers because it can model the full question-answer-context relation rather than only token overlap.
- The benchmark includes controlled hallucinations such as negation flips, entity replacements, number shifts, dosage changes, temporal contradictions, phrase drops, and unsupported appended claims.

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
