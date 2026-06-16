import argparse
import json
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch


def format_prompt(example):
    prompt = f"""Bạn là trợ lý y tế. Trả lời ngắn gọn và chính xác.
Câu hỏi: {example['question_vi']}
Câu trả lời: {example['answer_vi']}"""
    return {"text": prompt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", default="checkpoints/qlora")
    args = parser.parse_args()

    data = load_dataset("json", data_files=args.train)["train"]
    data = data.map(format_prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    tokenized = data.map(tokenize, batched=True, remove_columns=data.column_names)

    args_train = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized,
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
