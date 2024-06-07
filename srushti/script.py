import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from transformers import AutoModelForCausalLM, , BitsAndBytesConfig, TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import time
import json
import random
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto

access_token = "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"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False, token=access_token)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
model.to(device)

tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("srushtisingh/Ethical")  # Replace "your_data.jsonl" with your file path


def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["revised_answer"], truncation=True, padding="max_length",
                     max_length=60)


# Tokenize dataset
tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["prompt", "revised_answer"],
                                         desc="Tokenizing dataset")

# Split dataset into train and eval
train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42).values()

# Training Arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train and Evaluate
progress_bar = tqdm(range(training_args.num_train_epochs))
for epoch in progress_bar:
    trainer.train()
    results = trainer.evaluate()
    progress_bar.set_postfix(eval_loss=results["eval_loss"])

print(results)

trainer.model.push_to_hub("srushtisingh/EthicalSFT")
