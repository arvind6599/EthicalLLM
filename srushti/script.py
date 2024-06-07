import torch
from transformers import BitsAndBytesConfig, HfArgumentParser, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from transformers.adapters import AdapterType, AdapterConfig
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

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
    config=lora_config,
    quantization_config=quantization_config,
    device_map=device_map
)

adapter_config = AdapterConfig.load_adapter("text_task", config="houlsby", load_as="text_task", with_head=False)
model.add_adapter("text_task", AdapterType.text_task, config=adapter_config)

# model_for_classification = AutoModelForSequenceClassification.from_pretrained(model_name, config=lora_config,
# quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", model_max_length=100,
                                          padding="max_length", truncation=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model_for_classification.resize_token_embeddings(len(tokenizer))
model_for_classification.config.pad_token_id = model_for_classification.config.eos_token_id

access_token = "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"
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False,
# token=access_token) tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
# token=access_token) model.to(device)


# tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("srushtisingh/Ethical")  # Replace "your_data.jsonl" with your file path

# Tokenize Dataset
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["prompt"], text_target=examples["revised_answer"], truncation=True, padding="max_length", max_length=60),
    batched=True,
    remove_columns=["prompt", "revised_answer"]
)

# Tokenize dataset
# tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["prompt", "revised_answer"])

# Split dataset into train and eval
train_dataset, eval_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42).values()

# Training Arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    evaluation_strategy="steps",
    learning_rate=1.41e-5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=16,
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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
results = trainer.evaluate()
print(results)

trainer.model.push_to_hub("srushtisingh/EthicalSFT")
