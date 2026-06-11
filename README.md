# Vietnamese Medical RAG Reliability Lab

Hệ thống hỏi đáp y tế tiếng Việt có **RAG + hybrid retrieval + fine-tuning + hallucination detection + evaluation framework**. 

## Mục tiêu
Xây dựng hệ thống hỏi đáp tiếng Việt trong **miền tri thức y tế phổ thông** bằng **RAG + fine-tuning LLM**, sau đó đánh giá độ tin cậy bằng hallucination detection theo dạng `question + retrieved context + answer -> supported/hallucinated`.

## Checklist yêu cầu đề tài (này e ghi lại theo các đề bên môn DL trên trường)
### 1. Dữ liệu
- [x] Chọn domain: **Y tế phổ thông (Biomedical QA)**.
- [x] Thu thập tài liệu/đoạn văn để làm **Knowledge Base** (KB) → `data/kb_vi/`.
- [x] Tạo QA y tế tiếng Việt chuẩn hoá → `data/processed_vi/`.

### 2. Fine-tuning (LoRA/QLoRA)
- [x] Script QLoRA trên Colab Free: `src/finetune/qlora_train.py`.
- [x] Gợi ý model 1B–7B: `Qwen2.5-1.5B-Instruct`, `Llama-3.2-3B-Instruct`, `Gemma-2-2B`.

### 3. Pipeline RAG
- [x] Chunking + embedding + vector store (Chroma/FAISS) → `src/rag/build_kb.py`.
- [x] Retriever top-k + prompt template rõ ràng → `src/rag/query_rag.py`.
- [x] Hybrid retriever BM25 + dense score fusion → `src/rag/hybrid_retriever.py`.

### 4. Thực nghiệm 4 cấu hình
|                | LLM gốc | LLM fine-tuned |
|----------------|---------|----------------|
| **Không RAG**  | A       | C              |
| **Có RAG**     | B       | D              |

### 5. Đánh giá
- [x] BLEU, ROUGE-L, BERTScore → `src/eval/evaluate_metrics.py`.
- [x] Retrieval Recall@5 → `src/eval/evaluate_retrieval.py`.
- [x] Tạo benchmark hallucination có nhãn → `src/eval/create_hallucination_benchmark.py`.
- [x] Baseline hallucination judge offline → `src/eval/hallucination_baselines.py`.
- [x] Fine-tune Transformer judge cho hallucination detection → `src/finetune/train_hallucination_detector.py`.
- [x] Report Markdown + biểu đồ so sánh → `src/eval/generate_report.py`.
- [x] Human eval 50 câu (template trong `docs/human_eval_template.md`).

### 6. Demo
- [x] Demo Streamlit → `src/demo/app.py`.
