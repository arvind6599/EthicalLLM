import torch
from transformers import TextDataset, AutoConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM, \
    BitsAndBytesConfig
import torch
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import time
import json
import random
from sklearn.model_selection import train_test_split
import os

access_token = "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"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2, token=access_token)
model_for_classification = AutoModelForSequenceClassification.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    config=config,
    quantization_config=quantization_config,
    device_map=device_map
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", model_max_length=100,
                                          padding="max_length", truncation=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model_for_classification.resize_token_embeddings(len(tokenizer))
model_for_classification.config.pad_token_id = model_for_classification.config.eos_token_id

dataset = load_dataset("srushtisingh/Ethical")  # Replace "your_data.jsonl" with your file path


def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["revised_answer"], truncation=True, padding="max_length")


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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=16,
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

start = time.time()
# Train and Evaluate
progress_bar = tqdm(range(training_args.num_train_epochs))
for epoch in progress_bar:
    trainer.train()
    results = trainer.evaluate()
    progress_bar.set_postfix(eval_loss=results["eval_loss"])


print("Time taken for rm training:", time.time() - start)
print(results)

trainer.model.push_to_hub("srushtisingh/EthicalSFT")
