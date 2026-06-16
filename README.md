# Vietnamese Medical RAG Reliability Lab

Hệ thống hỏi đáp y tế tiếng Việt: 
- RAG 
- hybrid retrieval 
- fine-tuning 
- hallucination detection 
- evaluation framework 

## Mục tiêu
Xây dựng hệ thống hỏi đáp tiếng Việt trong miền tri thức y tế phổ thông bằng RAG + fine-tuning LLM

Sau đó đánh giá độ tin cậy bằng hallucination detection theo dạng question + retrieved context + answer -> supported/hallucinated.

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

## Cấu trúc project (dự kiến)
```
rag-llm-vietnamese/
├── data/
│   ├── kb/                 # -> Tài liệu domain (txt)
│   ├── kb_vi/              # -> Tài liệu y tế tiếng Việt
│   ├── raw/                # -> Dữ liệu gốc
│   ├── processed/          # -> QA đã chuẩn hoá
│   ├── processed_vi/       # -> QA y tế tiếng Việt
│   └── benchmarks/         # -> Hallucination benchmark sinh ra từ QA + KB
├── outputs/
│   ├── vectorstore/         # -> Chroma/FAISS
│   └── predictions/         # -> Dự đoán để đánh giá
├── src/
│   ├── config.yaml
│   ├── data/prepare_dataset.py
│   ├── rag/build_kb.py
│   ├── rag/query_rag.py
│   ├── rag/hybrid_retriever.py
│   ├── finetune/qlora_train.py
│   ├── finetune/train_hallucination_detector.py
│   ├── eval/evaluate_metrics.py
│   ├── eval/evaluate_retrieval.py
│   ├── eval/create_hallucination_benchmark.py
│   ├── eval/hallucination_baselines.py
│   ├── utils/jsonl.py
│   └── demo/app.py
├── docs/
│   ├── PROJECT_PLAN.md
│   └── human_eval_template.md
├── requirements.txt
└── .gitignore
```

## Hướng dẫn nhanh
### 1) Cài đặt
```bash
bash scripts/setup_env.sh --reset
```

Mặc định lệnh trên chỉ cài môi trường sạch cho demo, report và PhoBERT hallucination detector. Nếu cần build vector store RAG thì cài thêm:
```bash
INSTALL_RAG_DEPS=1 bash scripts/setup_env.sh --reset
```

QLoRA nên cài riêng trên Colab/Linux GPU:
```bash
./.venv/bin/python -m pip install -r requirements-qlora.txt
```

Nếu máy có alias `python=python3` hoặc `pip=pip3`, vẫn nên dùng `./.venv/bin/python -m pip ...` để chắc chắn package được cài vào đúng virtualenv.

Kiểm tra đúng môi trường:
```bash
bash scripts/check_env.sh
```

`sys.prefix` phải trỏ vào thư mục project `.venv`; nếu vẫn thấy `/Library/Frameworks/.../site-packages` trong traceback thì bạn đang không chạy bằng `.venv/bin/python`.

### 2) Chuẩn bị QA
- Đưa dữ liệu QA chuẩn hoá vào `data/processed_vi/bioasq_vi.jsonl`.
- Định dạng JSONL mỗi dòng:
```json
{"id":"...","question_vi":"...","answer_vi":"...","gold_docs":["doc_1","doc_2"]}
```

### 3) Build Knowledge Base
```bash
python src/rag/build_kb.py --kb_dir data/kb_vi --out_dir outputs/vectorstore_vi
```

### 4) Chạy RAG
```bash
python src/rag/query_rag.py --model Qwen/Qwen2.5-1.5B-Instruct --vectorstore outputs/vectorstore_vi --top_k 5
```

### 4.1) Test Hybrid Retrieval
```bash
python -m src.rag.hybrid_retriever \
  --kb_dir data/kb_vi \
  --vectorstore outputs/vectorstore_vi \
  --question "Bệnh Hirschsprung là rối loạn đơn gen hay đa yếu tố?" \
  --top_k 5
```

### 5) Fine-tune (Colab)
```bash
python src/finetune/qlora_train.py --model Qwen/Qwen2.5-1.5B-Instruct --train data/processed/train.jsonl
```

### 6) Đánh giá
```bash
python src/eval/evaluate_metrics.py --pred outputs/predictions/pred.jsonl --ref data/processed/test.jsonl
python src/eval/evaluate_retrieval.py --pred outputs/predictions/pred.jsonl --ref data/processed/test.jsonl
```

### 6.1) Tạo và đánh giá Hallucination Benchmark
```bash
python -m src.eval.create_hallucination_benchmark \
  --qa data/processed_vi/bioasq_vi.jsonl \
  --kb_dir data/kb_vi \
  --output data/benchmarks/hallucination_vi.jsonl

python -m src.eval.hallucination_baselines \
  --input data/benchmarks/hallucination_vi.jsonl \
  --output outputs/predictions/hallucination_baseline.jsonl
```

### 6.2) Fine-tune Transformer Hallucination Detector
```bash
python -m src.finetune.train_hallucination_detector \
  --train data/benchmarks/hallucination_vi.jsonl \
  --model vinai/phobert-base-v2 \
  --output_dir outputs/models/phobert_hallucination_detector

python -m src.eval.predict_hallucination_transformer \
  --input data/benchmarks/hallucination_vi.jsonl \
  --model outputs/models/phobert_hallucination_detector \
  --output outputs/predictions/phobert_hallucination_detector.jsonl
```

Nếu máy có TensorFlow global bị lỗi, chạy qua script để ép Transformers chỉ dùng PyTorch:
```bash
bash scripts/run_train_hallucination.sh \
  --train data/benchmarks/hallucination_vi.jsonl \
  --model vinai/phobert-base-v2 \
  --output_dir outputs/models/phobert_hallucination_detector
```

Nếu model đã tải rồi và muốn tránh gọi mạng lại:
```bash
bash scripts/run_train_hallucination.sh \
  --train data/benchmarks/hallucination_vi.jsonl \
  --model vinai/phobert-base-v2 \
  --output_dir outputs/models/phobert_hallucination_detector \
  --local_files_only
```

### 6.3) Sinh report + biểu đồ
```bash
python -m src.eval.generate_report
```

Smoke-test nhanh trên subset nhỏ:
```bash
python -m src.finetune.train_hallucination_detector \
  --train data/benchmarks/hallucination_vi.jsonl \
  --model vinai/phobert-base-v2 \
  --output_dir outputs/models/phobert_smoke \
  --max_samples 300 \
  --epochs 1
```

### 7) Demo
```bash
./.venv/bin/python -m streamlit run src/demo/app.py
```

Hoặc dùng script:
```bash
bash scripts/run_demo.sh
```

Chọn port khác nếu 8501 đang bận:
```bash
bash scripts/run_demo.sh 8502
```

## Lưu ý nộp bài
- Dataset + checkpoint: upload lên **HuggingFace Hub** và **Google Drive**.
- README nên ghi rõ: nguồn dữ liệu, license, cấu hình thực nghiệm, kết quả.

---
