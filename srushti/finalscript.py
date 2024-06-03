import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # the device to load the model onto

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

access_token = "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"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)

tokenizer.pad_token_id = tokenizer.eos_token_id

# num_prompts = len(dataset)

# Access the 'train' split
# train_dataset = dataset['train']

# Extract prompts from the 'train' split
# dataset_prompt = train_dataset['prompt']

def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = dataset['train']

dataset_input_ids = train_dataset['prompt']
model.to(device)

batch_size = 16
revised_data = []
dataset = load_dataset("fka/awesome-chatgpt-prompts")


# num_prompts = len(dataset)

# Access the 'train' split
# train_dataset = dataset['train']

# Extract prompts from the 'train' split
# dataset_prompt = train_dataset['prompt']

def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = dataset['train']

dataset_input_ids = train_dataset['prompt']
model.to(device)


def revision(principles_list):
    for i in tqdm(range(0, len(dataset_input_ids), batch_size)):
        batch_prompts = dataset_input_ids[i:i + batch_size]
        batch_input_ids = tokenizer.batch_encode_plus(batch_prompts, padding=True, return_tensors='pt')['input_ids'].to(
            device)
        # batch_input_ids = torch.stack(batch_input_ids)

        output = model.generate(batch_input_ids, max_new_tokens=50, num_beams=batch_size)
        output = output[:, batch_input_ids.shape[1]:]
        base_answers = tokenizer.batch_decode(output, skip_special_tokens=True)
        original_answers = base_answers.copy()

        for principle in principles_list:
            critique = [
                f"Revise the following response with respect to {principle}: '{base_answers}'. Avoid using any " \
                f"metacommentary such as 'A more respectful answer would be...' in the revised answer, " \
                f"instead directly provide a concise answer and finish it in 50 tokens. "
                for b in base_answers]
            new_inputs = tokenizer.batch_encode_plus(critique, return_tensors='pt')['prompt'].to(device)
            new_out = model.generate(new_inputs, max_new_tokens=50, num_beams=batch_size)
            new_out = new_out[:, new_inputs.shape[1]:]
            criticized_answers = tokenizer.batch_decode(new_out, skip_special_tokens=True)
            base_answers = criticized_answers

        with open("revised_datafile1.jsonl", "a") as f:
            for prompt, base_answer, revised_answer in zip(batch_prompts[i:i + batch_size], original_answers,
                                                           base_answers):
                data = {
                    "prompt": prompt,
                    "base_answer": base_answer,
                    "revised_answer": revised_answer,
                }
                print(revised_answer, "\n")
                f.write(json.dumps(data) + "\n")


principles_list = ["honesty", "prudence", "compassion", "justice", "humility", "respect"]
revision(principles_list)
