import argparse
import json


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    preds = load_jsonl(args.pred)
    refs = load_jsonl(args.ref)
    ref_map = {r["id"]: r.get("gold_docs", []) for r in refs}

    hits = 0
    total = 0
    for p in preds:
        gold = set(ref_map.get(p.get("id"), []))
        retrieved = set(p.get("retrieved", []))
        if not gold:
            continue
        total += 1
        if len(gold.intersection(retrieved)) > 0:
            hits += 1

    recall = hits / total if total > 0 else 0
    print(f"Recall@5: {recall:.4f} (hits={hits}, total={total})")


if __name__ == "__main__":
    main()
