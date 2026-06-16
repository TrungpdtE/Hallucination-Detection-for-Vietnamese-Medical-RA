import argparse
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

PROMPT_TEMPLATE = """
Bạn là trợ lý y tế. Dựa trên ngữ cảnh sau, hãy trả lời ngắn gọn và chính xác.

Ngữ cảnh:
{context}

Câu hỏi: {question}
Câu trả lời:
"""


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vectorstore", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embedding", default="intfloat/multilingual-e5-base")
    parser.add_argument("--question", default=None)
    parser.add_argument("--input", default=None, help="jsonl input for batch")
    parser.add_argument("--output", default="outputs/predictions/pred.jsonl")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.vectorstore)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embedding
    )
    collection = client.get_collection(name="kb", embedding_function=embed_fn)

    tokenizer, model = load_model(args.model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def answer_one(q):
        results = collection.query(query_texts=[q], n_results=args.top_k)
        context = "\n".join(results["documents"][0])
        prompt = PROMPT_TEMPLATE.format(context=context, question=q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.split("Câu trả lời:")[-1].strip(), results

    if args.question:
        ans, _ = answer_one(args.question)
        print(ans)
        return

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
            for line in f:
                item = json.loads(line)
                q = item["question_vi"]
                pred, results = answer_one(q)
                out.write(json.dumps({
                    "id": item.get("id"),
                    "question_vi": q,
                    "pred": pred,
                    "gold": item.get("answer_vi"),
                    "retrieved": results.get("ids", [[]])[0],
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
