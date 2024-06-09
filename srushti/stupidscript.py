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
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = model.config.eos_token_id

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def formatting_prompts_func(example):
    tokenizer.padding_side = 'right'
    output_texts = []
    for i in range(len(example['prompt'])):
        revised_answer = example['revised_answer'][i].replace('[PAD]', '').strip()
        text = f"### Prompt: {example['prompt'][i]}\n ### Response: {revised_answer}"
        output_texts.append(text)
    return output_texts


response_template = " ### Response:"
collator = DataCollatorForCompletionOnlyLM(tokenizer.encode(f"\n{response_template}", add_special_tokens = False)[2:],
                                           tokenizer=tokenizer)

cmd_args = [
    "--per_device_train_batch_size=20",
    "--output_dir=reward_modeling_consequences",
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
    ################
    # Config parsing
    ################
parser = HfArgumentParser((SFTConfig, ModelConfig))
reward_config, model_config = parser.parse_args_into_dataclasses(args=cmd_args)
reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
    )


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=reward_config,
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
