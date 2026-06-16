import argparse
import json
from evaluate import load


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
    ref_map = {r["id"]: r["answer_vi"] for r in refs}

    pred_texts, ref_texts = [], []
    for p in preds:
        if p.get("id") not in ref_map:
            continue
        pred_texts.append(p["pred"])
        ref_texts.append(ref_map[p["id"]])

    bleu = load("sacrebleu")
    rouge = load("rouge")
    bertscore = load("bertscore")

    bleu_res = bleu.compute(predictions=pred_texts, references=[[r] for r in ref_texts])
    rouge_res = rouge.compute(predictions=pred_texts, references=ref_texts)
    bert_res = bertscore.compute(predictions=pred_texts, references=ref_texts, lang="vi")

    print("BLEU:", bleu_res)
    print("ROUGE-L:", rouge_res["rougeL"])
    print("BERTScore (F1 mean):", sum(bert_res["f1"]) / len(bert_res["f1"]))


if __name__ == "__main__":
    main()
