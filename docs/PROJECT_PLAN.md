# Vietnamese Medical RAG Reliability Lab

## One-line CV pitch

Build a Vietnamese medical RAG system with hybrid retrieval, answer generation, hallucination detection, and a reproducible evaluation suite for retrieval quality, answer quality, factual grounding, latency, and cost.

## Why this project is stronger than a normal chatbot

Normal RAG demos usually stop at "retrieve top-k chunks and ask an LLM". This project turns your Vietnamese medical dataset into a reliability benchmark:

- Retrieval: BM25 vs dense embeddings vs hybrid retrieval.
- Generation: no-RAG LLM vs RAG LLM vs fine-tuned LLM vs fine-tuned RAG.
- Faithfulness: detect whether an answer is supported by the retrieved medical evidence.
- Operations: report latency, context length, and failure cases.
- Demo: Streamlit app with answer, citations, retrieval scores, and hallucination warning.

## Research direction

Recommended title:

**Hallucination Detection for Vietnamese Medical RAG Systems using Hybrid Retrieval and Transformer-based Judge Models**

This is attractive for AI/NLP interviews because it combines applied RAG engineering with a research-style evaluation problem. Recent work such as LettuceDetect shows that hallucination detection for RAG can be framed as context-question-answer verification, and multilingual/domain-specific variants are still underexplored.

## Dataset layout

Current local data can be used directly:

- `data/kb_vi/*.txt`: Vietnamese medical evidence documents.
- `data/processed_vi/bioasq_vi.jsonl`: Vietnamese QA pairs.
- `data/processed/*.jsonl`: English QA split, useful for ablation or translation comparison.

Recommended benchmark schema:

```json
{
  "id": "55031181e9bde69634000014:supported",
  "source_id": "55031181e9bde69634000014",
  "question": "Bệnh Hirschsprung là rối loạn đơn gen hay đa yếu tố?",
  "context": "...retrieved or gold evidence...",
  "answer": "...candidate answer...",
  "label": "supported",
  "hallucination_type": "none",
  "gold_docs": ["55031181e9bde69634000014.txt"]
}
```

## Milestones

### Phase 1: Strong baseline

- Build Chroma vector store over `data/kb_vi`.
- Run dense retrieval with multilingual E5.
- Add BM25 and hybrid retrieval.
- Measure Recall@1/3/5/10 and MRR.
- Produce an error table: missing evidence, wrong top-1, lexical mismatch, entity mismatch.

### Phase 2: RAG answer generation

- Compare:
  - no-RAG base LLM
  - dense RAG
  - hybrid RAG
  - hybrid RAG with reranking
- Save all predictions to JSONL.
- Evaluate BLEU, ROUGE-L, BERTScore, exact evidence hit, answerability behavior.

### Phase 3: Hallucination benchmark

- Generate supported examples from gold answers.
- Generate hallucinated examples with controlled corruptions:
  - number shift
  - negation flip
  - relation reversal
  - unsupported phrase append
  - entity or phrase drop
- Manually review 200-500 examples for a clean test set.

### Phase 4: Judge models

- Baseline 1: lexical/number consistency detector.
- Baseline 2: multilingual NLI or cross-encoder judge.
- Main model: fine-tune PhoBERT/ViHealthBERT/ModernBERT-style encoder for binary classification.
- Stretch goal: token-level unsupported span detection.

### Phase 5: Demo and report

- Streamlit demo:
  - question input
  - retrieved evidence with source file and score
  - generated answer
  - hallucination risk label
  - highlighted unsupported numbers/terms
- Final report:
  - dataset statistics
  - methods
  - experiments
  - ablations
  - failure analysis
  - ethical and medical safety note

## Commands

Build Vietnamese vector store:

```bash
python src/rag/build_kb.py \
  --kb_dir data/kb_vi \
  --out_dir outputs/vectorstore_vi \
  --embedding intfloat/multilingual-e5-base
```

Create hallucination benchmark:

```bash
python -m src.eval.create_hallucination_benchmark \
  --qa data/processed_vi/bioasq_vi.jsonl \
  --kb_dir data/kb_vi \
  --output data/benchmarks/hallucination_vi.jsonl
```

Run lexical/number baseline judge:

```bash
python -m src.eval.hallucination_baselines \
  --input data/benchmarks/hallucination_vi.jsonl \
  --output outputs/predictions/hallucination_baseline.jsonl
```

Fine-tune Transformer judge:

```bash
python -m src.finetune.train_hallucination_detector \
  --train data/benchmarks/hallucination_vi.jsonl \
  --model vinai/phobert-base-v2 \
  --output_dir outputs/models/phobert_hallucination_detector
```

Test hybrid retrieval:

```bash
python -m src.rag.hybrid_retriever \
  --kb_dir data/kb_vi \
  --vectorstore outputs/vectorstore_vi \
  --question "Bệnh Hirschsprung là rối loạn đơn gen hay đa yếu tố?" \
  --top_k 5
```

## Interview talking points

- Why hybrid retrieval helps Vietnamese medical text: BM25 preserves exact medical terms, abbreviations, and numbers; dense embeddings improve semantic recall.
- Why answer metrics are insufficient: BLEU/ROUGE can reward fluent overlap but miss unsupported factual claims.
- Why hallucination detection matters: medical RAG needs evidence-grounded answers and refusal behavior when evidence is missing.
- What you built beyond a demo: benchmark construction, reproducible evaluation, ablations, and failure analysis.

## Target results to report

Use a table like this after experiments:

| System | Recall@5 | MRR | BERTScore F1 | Hallucination F1 | Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | TBD | TBD | - | - | TBD |
| Dense E5 | TBD | TBD | - | - | TBD |
| Hybrid | TBD | TBD | - | - | TBD |
| Dense RAG | TBD | TBD | TBD | TBD | TBD |
| Hybrid RAG | TBD | TBD | TBD | TBD | TBD |
| Hybrid RAG + Judge | TBD | TBD | TBD | TBD | TBD |
