import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import argparse
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

from src.utils.jsonl import get_answer, get_gold_docs, get_question, read_jsonl, write_jsonl


NEGATIONS = [
    (" có ", " không có "),
    (" là ", " không phải là "),
    (" tăng ", " giảm "),
    (" giảm ", " tăng "),
    (" cao ", " thấp "),
    (" thấp ", " cao "),
    (" chủ yếu ", " hiếm khi "),
    (" predominantly ", " rarely "),
    (" is ", " is not "),
    (" are ", " are not "),
    (" increases ", " decreases "),
    (" decreases ", " increases "),
]

ENTITY_REPLACEMENTS = [
    ("trastuzumab", "tamoxifen"),
    ("tamoxifen", "trastuzumab"),
    ("aspirin", "warfarin"),
    ("warfarin", "aspirin"),
    ("insulin", "metformin"),
    ("metformin", "insulin"),
    ("ung thư vú", "ung thư phổi"),
    ("ung thư phổi", "ung thư vú"),
    ("bệnh nhân", "người khỏe mạnh"),
    ("phụ nữ", "nam giới"),
]

TEMPORAL_REPLACEMENTS = [
    ("trước", "sau"),
    ("sau", "trước"),
    ("cấp tính", "mạn tính"),
    ("mạn tính", "cấp tính"),
    ("ngắn hạn", "dài hạn"),
    ("dài hạn", "ngắn hạn"),
]

UNSUPPORTED_CLAIMS = [
    " Điều này chứng minh phương pháp điều trị luôn an toàn tuyệt đối.",
    " Tài liệu cũng khuyến cáo tăng gấp đôi liều cho mọi bệnh nhân.",
    " Kết luận này áp dụng cho cả trẻ em và phụ nữ mang thai.",
    " Không cần theo dõi thêm sau khi bắt đầu điều trị.",
]


def read_doc(kb_dir: str, filename: str) -> str:
    path = Path(kb_dir) / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def short_context(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def corrupt_answer(answer: str, rng: random.Random) -> Tuple[str, str]:
    padded = f" {answer} "
    strategies = []

    negations = [(src, tgt) for src, tgt in NEGATIONS if src in padded]
    if negations:
        strategies.append("negation")

    entities = [(src, tgt) for src, tgt in ENTITY_REPLACEMENTS if src.lower() in answer.lower()]
    if entities:
        strategies.append("entity_replacement")

    temporals = [(src, tgt) for src, tgt in TEMPORAL_REPLACEMENTS if src in answer.lower()]
    if temporals:
        strategies.append("temporal_contradiction")

    numbers = re.findall(r"\d+(?:[.,]\d+)?", answer)
    if numbers:
        strategies.append("number_shift")
        strategies.append("dosage_modification")

    tokens = answer.split()
    if len(tokens) > 8:
        strategies.append("entity_or_phrase_drop")

    strategies.append("unsupported_append")
    strategy = rng.choice(strategies)

    if strategy == "negation":
        src, tgt = rng.choice(negations)
        return padded.replace(src, tgt, 1).strip(), f"replace:{src.strip()}->{tgt.strip()}"

    if strategy == "entity_replacement":
        src, tgt = rng.choice(entities)
        return re.sub(re.escape(src), tgt, answer, count=1, flags=re.IGNORECASE), f"entity:{src}->{tgt}"

    if strategy == "temporal_contradiction":
        src, tgt = rng.choice(temporals)
        return re.sub(re.escape(src), tgt, answer, count=1, flags=re.IGNORECASE), f"temporal:{src}->{tgt}"

    if strategy == "number_shift":
        n = rng.choice(numbers)
        replacement = str(int(float(n.replace(",", "."))) + rng.choice([1, 2, 5]))
        return answer.replace(n, replacement, 1), "number_shift"

    if strategy == "dosage_modification":
        n = rng.choice(numbers)
        try:
            value = float(n.replace(",", "."))
            replacement = str(max(1, int(round(value * rng.choice([2, 3, 10])))))
        except ValueError:
            replacement = str(int(float(n.replace(",", "."))) + 10)
        return answer.replace(n, replacement, 1), "dosage_modification"

    if strategy == "entity_or_phrase_drop":
        drop_at = rng.randrange(0, len(tokens) - 3)
        del tokens[drop_at : drop_at + 3]
        return " ".join(tokens), "entity_or_phrase_drop"

    return answer + rng.choice(UNSUPPORTED_CLAIMS), "unsupported_append"


def build_examples(rows: Iterable[Dict], kb_dir: str, max_context_chars: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    examples = []
    for row in tqdm(list(rows), desc="Building hallucination benchmark"):
        question = get_question(row)
        answer = get_answer(row)
        gold_docs = get_gold_docs(row)
        if not gold_docs and row.get("id"):
            gold_docs = [f"{row['id']}.txt"]
        if not question or not answer or not gold_docs:
            continue

        context_parts = [read_doc(kb_dir, doc) for doc in gold_docs]
        context = short_context("\n".join(part for part in context_parts if part), max_context_chars)
        if not context:
            continue

        base = {
            "source_id": row.get("id"),
            "question": question,
            "context": context,
            "gold_docs": gold_docs,
        }
        examples.append(
            {
                **base,
                "id": f"{row.get('id')}:supported",
                "answer": answer,
                "label": "supported",
                "hallucination_type": "none",
            }
        )
        corrupted, corruption_type = corrupt_answer(answer, rng)
        examples.append(
            {
                **base,
                "id": f"{row.get('id')}:hallucinated",
                "answer": corrupted,
                "label": "hallucinated",
                "hallucination_type": corruption_type,
            }
        )

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True, help="QA JSONL, e.g. data/processed_vi/bioasq_vi.jsonl")
    parser.add_argument("--kb_dir", required=True, help="Folder containing evidence txt files")
    parser.add_argument("--output", default="data/benchmarks/hallucination_vi.jsonl")
    parser.add_argument("--max_context_chars", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.qa)
    examples = build_examples(rows, args.kb_dir, args.max_context_chars, args.seed)
    write_jsonl(args.output, examples)
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
