from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
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

CUDA_LAUNCH_BLOCKING=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
access_token = "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"

dataset = load_dataset("srushtisingh/Ethical", split="train")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token,
                                             quantization_config=quantization_config,
                                             device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model.config.pad_token_id = model.config.eos_token_id

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Prompt: {example['prompt'][i]}\n ### Response: {example['revised_answer'][i]}"
        output_texts.append(text)
    return output_texts


response_template = " ### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="/tmp",
                   ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config
)
start = time.time()
print("Training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
# trainer.train()
print("Completed!")
print("Time taken for sft training:", time.time() - start)
print("Pushing to hub...")
trainer.model.push_to_hub("srushtisingh/EthicalSFT")
# trainer.train()