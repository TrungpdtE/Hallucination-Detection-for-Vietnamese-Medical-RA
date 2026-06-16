import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_question(row: Dict[str, Any]) -> Optional[str]:
    return row.get("question_vi") or row.get("question") or row.get("query")


def get_answer(row: Dict[str, Any]) -> Optional[str]:
    return row.get("answer_vi") or row.get("answer") or row.get("gold") or row.get("reference")


def get_gold_docs(row: Dict[str, Any]) -> List[str]:
    docs = row.get("gold_docs") or row.get("gold_doc_ids") or []
    if isinstance(docs, str):
        return [docs]
    return list(docs)
