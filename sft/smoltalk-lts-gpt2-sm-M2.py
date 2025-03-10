import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
import torch
import multiprocessing
import random
import os

def chatml_tokenize(batch, tokenizer):
    texts = []
    for messages in batch["messages"]:
        chat = ""
        for msg in messages:
            if msg["role"] == "user":
                chat += "<|user|> " + msg["content"].strip() + " " + tokenizer.eos_token + " "
            elif msg["role"] == "assistant":
                chat += "<|assistant|> " + msg["content"].strip() + " " + tokenizer.eos_token + " "
        texts.append(chat.strip())
    return tokenizer(texts, padding=False, truncation=False)

def base_model_eval(question, tokenizer, model, device):
    encoded = tokenizer(question, return_tensors="pt").to(device)
    generated = model.generate(**encoded, max_new_tokens=20)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def chatml_eval(question, tokenizer, model, device):
    formatted_prompt = f"<|user|> {question} <|assistant|>"
    encoded = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    generated = model.generate(**encoded, max_new_tokens=100)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    wandb.login()

    dataset = load_dataset("HuggingFaceTB/smoltalk", 'all')

    tokenizer = AutoTokenizer.from_pretrained("microsoft/lts-gpt2-sm")
    model = AutoModelForCausalLM.from_pretrained("microsoft/lts-gpt2-sm", subfolder="gpt2_538d4b101df48595a935d90dbf4a7fb2ac09ac01")

    num_proc = multiprocessing.cpu_count()
    tokenized_train = dataset["train"].map(
        lambda batch: chatml_tokenize(batch, tokenizer),
        batched=True, batch_size=1000, num_proc=num_proc, remove_columns=["messages"]
    )
    tokenized_test = dataset["test"].map(
        lambda batch: chatml_tokenize(batch, tokenizer),
        batched=True, batch_size=1000, num_proc=num_proc, remove_columns=["messages"]
    )

    device = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    special_tokens = ["<|user|>", "<|assistant|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = model.to("cpu")
    model.resize_token_embeddings(len(tokenizer)) # MPS doesn't support model.resize_token_embeddings
    model = model.to(device)

    random_indices = random.sample(range(len(tokenized_test)), 50)
    sampled_eval_dataset = tokenized_test.select(random_indices)

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = SFTConfig(
        output_dir="./trainer_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=len(tokenized_train) // 100,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=10,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        dataloader_num_workers=1,
        gradient_checkpointing=True,
        optim="adafactor",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        run_name="m2-sm"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test.select(random_indices),
        data_collator=data_collator,
    )

    print("BEFORE TRAINING (Raw model):")
    print(base_model_eval("The capital of France is", tokenizer, model, device))
    print('\n...\n')
    print(base_model_eval("What is the capital of France?", tokenizer, model, device))

    trainer.train()

    print("\nAFTER TRAINING (ChatML-formatted):")
    print(chatml_eval("The capital of France is", tokenizer, model, device))
    print('\n...\n')
    print(chatml_eval("What is the capital of France?", tokenizer, model, device))
