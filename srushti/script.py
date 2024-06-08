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
from trl import ModelConfig, SFTTrainer
import transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
access_token = "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"

# new code:
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2, use_auth_token=access_token)

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

actual_model = AutoModelForSequenceClassification.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", token=access_token)

tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
# model_for_classification = AutoModelForSequenceClassification.from_pretrained(model_name, config=lora_config,
# quantization_config=quantization_config)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", model_max_length=100,
#                                       padding="max_length", truncation=True)
'''
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=access_token,
                                              padding="max_length", truncation=True)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", model_max_length=100,
                                          padding="max_length", truncation=True, token=access_token)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    actual_model.resize_token_embeddings(len(tokenizer))
actual_model.config.pad_token_id = actual_model.config.eos_token_id
'''
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False,
# token=access_token) tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
# token=access_token) model.to(device)


# tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("srushtisingh/Ethical")

dataset_train = dataset["train"]

'''
def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["revised_answer"], truncation=True, padding="max_length",
                     max_length=60)


tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)
'''


def preprocess_function(examples, tokenizer=tokenizer):
    new_examples = {
        "input_ids_prompt": [],
        "attention_mask_prompt": [],
        "input_ids_revised": [],
        "attention_mask_revised": [],
    }
    for prompt, answers in zip(examples["prompt"], examples["revised_answer"]):
        tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length")
        tokenized_rejected = tokenizer(answers, truncation=True, padding="max_length")
        new_examples["input_ids_prompt"].append(tokenized_prompt["input_ids"])
        new_examples["attention_mask_prompt"].append(tokenized_prompt["attention_mask"])
        new_examples["input_ids_revised"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_revised"].append(tokenized_rejected["attention_mask"])

    return new_examples


def preprocess_function1(examples):
    inputs = [str(example) for example in examples["prompt"]]
    targets = [str(example) for example in examples["revised_answer"]]

    # Tokenize the prompts
    tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=100)
    tokenized_targets = tokenizer(targets, truncation=True, padding="max_length", max_length=100)

    return {
        "input_ids_prompt": tokenized_inputs["input_ids"],
        "attention_mask_prompt": tokenized_inputs["attention_mask"],
        "input_ids_revised": tokenized_targets["input_ids"],
        "attention_mask_revised": tokenized_targets["attention_mask"]
    }


training_dataset = dataset_train.map(
    preprocess_function1,
    batched=True,
    num_proc=4
)
# Tokenize dataset
# tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["prompt", "revised_answer"])

cmd_args = [
    "--per_device_train_batch_size=20",
    "--output_dir=sft_model",
    "--num_train_epochs=2",
    "--gradient_accumulation_steps=16",
    "--gradient_checkpointing=True",
    "--learning_rate=1.41e-5",
    "--report_to=tensorboard",
    "--remove_unused_columns=False",
    "--optim=adamw_torch",
    "--logging_steps=10",
    "--evaluation_strategy=steps",
    "--max_length=100",
    "--save_total_limit=1",
    "--load_best_model_at_end=True",
    "--fp16=True",
    "--fp16_opt_level=O1",
]
# parser = HfArgumentParser((SFTConfig, ModelConfig))
# sft_config, model_config = parser.parse_args_into_dataclasses(args=cmd_args)
# sft_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
'''
model_kwargs = dict(
    trust_remote_code=model_config.trust_remote_code,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
)'''

# Training Arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    evaluation_strategy="steps",
    learning_rate=1.41e-5,
    per_device_train_batch_size=16,
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
trainer = SFTTrainer(
    model=actual_model,
    args=training_args,
    train_dataset=training_dataset,
    tokenizer=tokenizer,
    formatting_func=preprocess_function1,
    max_seq_length=50
)
start = time.time()
print("Training...")
trainer.train()
print("Completed!")
print("Time taken for sft training:", time.time() - start)
print("Pushing to hub...")
trainer.model.push_to_hub("srushtisingh/EthicalSFT")
