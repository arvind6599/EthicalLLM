import torch
from transformers import BitsAndBytesConfig, HfArgumentParser, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
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
from trl import SFTConfig
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
config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=access_token)

# lora config
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.01
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
    config=config,
    quantization_config=quantization_config,
    device_map=device_map, token=access_token
)
model_for_classification = get_peft_model(model_for_classification, lora_config)

# tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
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
'''
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", model_max_length=100,
                                          padding="max_length", truncation=True, token=access_token)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model_for_classification.resize_token_embeddings(len(tokenizer))
model_for_classification.config.pad_token_id = model_for_classification.config.eos_token_id

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_cache=False,
# token=access_token) tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
# token=access_token) model.to(device)

# tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("srushtisingh/Ethical")

dataset_train = dataset["train"]
print(len(dataset_train))

'''
def tokenize_function(examples):
    return tokenizer(examples["prompt"], text_target=examples["revised_answer"], truncation=True, padding="max_length",
                     max_length=60)


tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)
'''
train_data = dataset_train.select(
    [i for i in range(len(dataset["train"])) if i % 10 != 0])  # Use 90% of the data for training
val_data = dataset_train.select(
    [i for i in range(len(dataset["train"])) if i % 10 == 0])  # Use 10% of the data for validation
print(len(train_data))
print(len(val_data))

'''
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], return_tensors='pt', padding='max_length', max_length=60,
                       truncation=True)
    labels = tokenizer(examples['revised_answer'], return_tensors='pt', padding='max_length', max_length=60,
                       truncation=True)
    return {'input_ids': inputs['input_ids'], 'labels': labels['input_ids']}


# Apply tokenization to the datasets
train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
print(len(train_data))
print(len(val_data))
print(train_data[0])

'''


def preprocess_function(examples, tokenizer=tokenizer):
    new_examples = {
        "input_ids_prompt": [],
        "attention_mask_prompt": [],
        "input_ids_revised": [],
        "attention_mask_revised": [],
    }
    tokenizer.truncation_side = "left"
    for prompt, answers in zip(examples["prompt"], examples["revised_answer"]):
        tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length", max_length=60)
        tokenized_rejected = tokenizer(answers, truncation=True, padding="max_length", max_length=60)
        new_examples["input_ids_prompt"].append(tokenized_prompt["input_ids"])
        new_examples["attention_mask_prompt"].append(tokenized_prompt["attention_mask"])
        new_examples["input_ids_revised"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_revised"].append(tokenized_rejected["attention_mask"])

    return new_examples


def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["prompt"])):
        instruction = examples["prompt"][i]
        response = examples["revised_answer"][i]
        text = f'''
        Prompt: {instruction}
        
        Response:{response}
        '''
        output_text.append(text)
    return output_text

'''
train_data = train_data.map(
    preprocess_function,
    batched=True,
    num_proc=4
)

val_data = val_data.map(
    preprocess_function,
    batched=True,
    num_proc=4
)'''
data_collator = DataCollatorWithPadding(tokenizer)
print(train_data[0])
# print(len(training_dataset))
# print(training_dataset[38])
# Tokenize dataset
# tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["prompt", "revised_answer"])
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=16,
    fp16=True,
    logging_steps=10,
    save_total_limit=1,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
)

SFTConfig(output_dir="/tmp")

# Initialize Trainer
trainer = SFTTrainer(
    model=model_for_classification,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    eval_dataset=val_data,
    data_collator=data_collator,
    formatting_func=formatting_prompts_func
)

start = time.time()
print("Training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")

try:
    trainer.evaluate()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during evaluation: {e}")
# trainer.train()
print("Completed!")
print("Time taken for sft training:", time.time() - start)
print("Pushing to hub...")
trainer.model.push_to_hub("srushtisingh/EthicalSFT")
