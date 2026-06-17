import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.eval.hallucination_baselines import predict as baseline_predict
from src.eval.hallucination_input import format_hallucination_input
from src.eval.metrics import compute_classification_metrics, confusion_matrix, label_counts
from src.utils.jsonl import read_jsonl


BENCHMARK_PATH = ROOT / "data/benchmarks/hallucination_vi.jsonl"
BASELINE_PATH = ROOT / "outputs/predictions/hallucination_baseline.jsonl"
PHOBERT_PATH = ROOT / "outputs/predictions/phobert_hallucination_detector.jsonl"
PHOBERT_MODEL_DIR = ROOT / "outputs/models/phobert_smoke"


st.set_page_config(page_title="Vietnamese Medical RAG Reliability", layout="wide")


@st.cache_data(show_spinner=False)
def load_rows(path: str):
    return read_jsonl(path) if Path(path).exists() else []


@st.cache_resource(show_spinner=False)
def load_phobert_model(model_dir: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def phobert_predict(row):
    import torch

    tokenizer, model, device = load_phobert_model(str(PHOBERT_MODEL_DIR))
    max_len = min(
        tokenizer.model_max_length,
        model.config.max_position_embeddings - 2
    )
    inputs = tokenizer(
        format_hallucination_input(row),
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    pred_id = int(probs.argmax().item())
    label = {0: "supported", 1: "hallucinated"}.get(pred_id, str(pred_id))
    return label, float(probs[pred_id].item())


def metrics_frame(rows, system_name):
    metrics = compute_classification_metrics(rows)
    return pd.DataFrame(
        [
            {"system": system_name, "metric": "accuracy", "score": metrics["accuracy"]},
            {"system": system_name, "metric": "macro_f1", "score": metrics["macro_f1"]},
            {"system": system_name, "metric": "supported_f1", "score": metrics["supported_f1"]},
            {"system": system_name, "metric": "hallucinated_f1", "score": metrics["hallucinated_f1"]},
        ]
    )


st.title("Vietnamese Medical RAG Reliability Lab")
st.caption("Hallucination detection dashboard for Vietnamese medical RAG: question + answer + context -> supported or hallucinated.")

benchmark_rows = load_rows(str(BENCHMARK_PATH))
baseline_rows = load_rows(str(BASELINE_PATH))
phobert_rows = load_rows(str(PHOBERT_PATH))

tab_demo, tab_compare, tab_data, tab_report = st.tabs(["Demo", "So sánh", "Dữ liệu", "Report"])

with tab_demo:
    if not benchmark_rows:
        st.warning("Chưa có benchmark. Chạy `python -m src.eval.create_hallucination_benchmark` trước.")
    else:
        left, right = st.columns([0.42, 0.58])
        with left:
            label_filter = st.selectbox("Nhãn", ["all", "supported", "hallucinated"])
            filtered = [row for row in benchmark_rows if label_filter == "all" or row.get("label") == label_filter]
            idx = st.slider("Mẫu", 0, max(len(filtered) - 1, 0), 0)
            threshold = st.slider("Baseline threshold", 0.0, 1.0, 0.68, 0.01)
            row = filtered[idx]

            st.metric("Gold label", row.get("label", "unknown"))
            st.write("**Question**")
            st.write(row.get("question", ""))
            st.write("**Candidate answer**")
            st.write(row.get("answer", ""))

            baseline = baseline_predict(row, threshold)
            st.write("**Baseline judge**")
            st.json(
                {
                    "pred_label": baseline["pred_label"],
                    "support_score": baseline["support_score"],
                    "features": baseline["features"],
                }
            )

            if PHOBERT_MODEL_DIR.exists():
                pred_label, confidence = phobert_predict(row)
                st.write("**PhoBERT judge**")
                st.json({"pred_label": pred_label, "confidence": round(confidence, 4)})
            else:
                st.info("Chưa có checkpoint PhoBERT local. Train model để bật PhoBERT judge trong demo.")

        with right:
            st.write("**Evidence context**")
            st.text_area("Context", row.get("context", ""), height=520, label_visibility="collapsed")
            st.write("**Gold docs**")
            st.write(row.get("gold_docs", []))

with tab_compare:
    frames = []
    if baseline_rows:
        frames.append(metrics_frame(baseline_rows, "Baseline"))
    if phobert_rows:
        frames.append(metrics_frame(phobert_rows, "PhoBERT"))

    if not frames:
        st.warning("Chưa có prediction. Chạy baseline hoặc PhoBERT prediction trước.")
    else:
        metrics_df = pd.concat(frames, ignore_index=True)
        pivot = metrics_df.pivot(index="metric", columns="system", values="score")
        st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)
        st.bar_chart(metrics_df, x="metric", y="score", color="system")

        cols = st.columns(len(frames))
        for col, (system_name, rows) in zip(cols, [("Baseline", baseline_rows), ("PhoBERT", phobert_rows)]):
            if rows:
                with col:
                    st.write(f"**Confusion matrix - {system_name}**")
                    matrix = confusion_matrix(rows)
                    st.dataframe(pd.DataFrame(matrix).T, use_container_width=True)

with tab_data:
    if not benchmark_rows:
        st.warning("Benchmark chưa tồn tại.")
    else:
        st.write("**Dataset statistics**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Examples", len(benchmark_rows))
        c2.metric("Supported", label_counts(benchmark_rows).get("supported", 0))
        c3.metric("Hallucinated", label_counts(benchmark_rows).get("hallucinated", 0))
        type_counts = pd.Series(label_counts(benchmark_rows, "hallucination_type")).sort_values(ascending=False)
        st.bar_chart(type_counts)
        preview = pd.DataFrame(benchmark_rows)[["id", "label", "hallucination_type", "question", "answer"]].head(100)
        st.dataframe(preview, use_container_width=True, height=420)

with tab_report:
    report_path = ROOT / "reports/hallucination_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
        for chart in sorted((ROOT / "reports").glob("*.png")):
            st.image(str(chart), caption=chart.name)
    else:
        st.info("Chưa có report. Chạy `python -m src.eval.generate_report` để sinh Markdown và biểu đồ.")
