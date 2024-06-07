import torch
from transformers import BitsAndBytesConfig, HfArgumentParser, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import time
import json
import random
from sklearn.model_selection import train_test_split
from peft import (
    LoraConfig,
    TaskType,
)
import os
import time
from peft import get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
access_token = "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"

# new code:
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
# config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2)

# lora config
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none'
)

model_for_classification = AutoModelForSequenceClassification.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=quantization_config,
    device_map=device_map, token=access_token
)
model_for_classification = get_peft_model(model_for_classification, lora_config)

# model_for_classification = AutoModelForSequenceClassification.from_pretrained(model_name, config=lora_config,
# quantization_config=quantization_config)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", model_max_length=100,
#                                       padding="max_length", truncation=True)

try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=access_token,
                                              padding="max_length", truncation=True)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model_for_classification.resize_token_embeddings(len(tokenizer))
model_for_classification.config.pad_token_id = model_for_classification.config.eos_token_id

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False,
# token=access_token) tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
# token=access_token) model.to(device)


# tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("srushtisingh/Ethical")  # Replace "your_data.jsonl" with your file path


def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["revised_answer"], truncation=True, padding="max_length",
                     max_length=60)


tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["prompt", "revised_answer"])

# Tokenize dataset
# tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["prompt", "revised_answer"])

# Split dataset into train and eval

#print(len(tokenized_dataset["train"][0]))  # This should match the expected batch size
# Get the first batch of the tokenized dataset
#first_batch = tokenized_dataset["train"][0]
#print(first_batch["input_ids"].shape)  # This will give you the (batch_size, sequence_length)
#print(first_batch["labels"].shape)  # This will give you the (batch_size,)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    evaluation_strategy="steps",
    learning_rate=1.41e-5,
    per_device_train_batch_size=960,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=960,
    fp16=True,
    logging_steps=10,
    save_total_limit=1,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model_for_classification,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.model.push_to_hub("srushtisingh/EthicalSFT")
