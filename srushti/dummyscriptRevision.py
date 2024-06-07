import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datasets import Dataset, concatenate_datasets

pattern = r"Human:(.*?)\n\n"


def extract_human_prompts(transcript):
    match = re.search(pattern, transcript, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_human_prompts_parallel(batch_transcripts):
    with ThreadPoolExecutor() as executor:
        prompts = list(executor.map(extract_human_prompts, batch_transcripts))
    return [prompt for prompt in prompts if prompt is not None]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto

access_token = "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"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)

tokenizer.pad_token_id = tokenizer.eos_token_id

batch_size = 16
revised_data = []
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
num_prompts = len(dataset)

# Access the 'train' split
train_dataset = dataset['train']

# Extract prompts from the 'train' split
dataset_prompt = train_dataset['transcript']
dataset_prompt = dataset_prompt[:40]
# dataset_prompt = extract_human_prompts(dataset_prompt, batch_size)

model.to(device)


class RevisionModel(nn.Module):
    def __init__(self, model, tokenizer, principles_list):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.principles_list = principles_list

    def forward(self, prompts, base_answers):
        revised_answers = []
        for prompt, base_answer in zip(prompts, base_answers):
            original_answer = base_answer
            for principle in self.principles_list:
                critique = (
                    f"Criticize the following response for the given prompt in terms of {principle} principle and revise it: 'prompt:{prompt}\nbase answer:{base_answer}'. "
                    f"Provide a concise, coherent and complete answer within approximately 50 tokens (or less if you can), ensuring it is a "
                    f"complete, meaningful, direct and finished response."
                    f"Remove additional commentary or explanations from the revised answer such as 'A more respectful answer would be...', or 'Revised response with"
                    f" respect to...' and provide only the terminal output in the revised answer."
                    f" Remove the token count from the revised answer such as '(50 tokens)'."
                    f" Remove any '\n\n Revised response: \n\n' from the revised response."
                )
                new_inputs = self.tokenizer.encode(critique, return_tensors='pt').to(self.model.device)
                with torch.no_grad():
                    new_out = self.model.generate(new_inputs, max_new_tokens=50)
                new_out = new_out[:, len(new_inputs[0]):]
                criticized_answer = self.tokenizer.decode(new_out[0], skip_special_tokens=True)
                base_answer = criticized_answer
            revised_answers.append(criticized_answer)
        return revised_answers


def revision(principles_list):
    revised_data = []

    revision_model = RevisionModel(model, tokenizer, principles_list).to(device)
    # revision_model = nn.DataParallel(revision_model)

    for i in tqdm(range(0, len(dataset_prompt), batch_size)):
        batch_transcripts = dataset_prompt[i:i + batch_size]
        batch_prompts = extract_human_prompts_parallel(batch_transcripts)

        input_ids = tokenizer.batch_encode_plus(batch_prompts, return_tensors='pt', padding=True)
        input_ids = input_ids.input_ids.to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50)
        output = output[:, input_ids.shape[1]:]

        base_answers = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

        # revision_model = RevisionModel(model, tokenizer, principles_list).to(device)
        # revision_model = nn.DataParallel(revision_model)

        revised_answers = revision_model(batch_prompts, base_answers)

        for j, prompt in enumerate(batch_prompts):
            revised_data.append({
                "prompt": prompt,
                "base_answer": base_answers[j],
                "revised_answer": revised_answers[j]
            })

        torch.cuda.empty_cache()

    with open("revised_datafile_redteam.jsonl", "w") as f:
        for data in revised_data:
            json_line = json.dumps(data)
            f.write(json_line + "\n")


def load_jsonl_data_in_batches(filepath, batch_size=32):
    with open(filepath, 'r') as f:
        batch = []
        for i, line in enumerate(f):
            batch.append(json.loads(line))
            if (i + 1) % batch_size == 0:
                yield batch
                batch = []
        if batch:
            yield batch


def convert_to_dataframe(data):
    df = pd.DataFrame(data)
    return df


def prepare_data_for_hf(df):
    hf_data = {
        "prompt": df["prompt"].tolist(),
        "revised_answer": df["revised_answer"].tolist()
    }
    return hf_data


def convert_to_hf_dataset(data):
    hf_dataset = Dataset.from_dict(data)
    return hf_dataset


def upload_to_hf_hub(dataset, repo_name):
    dataset.push_to_hub(repo_name)


def convert_and_upload_to_hf_hub(filepath, repo_name, batch_size=1000):
    datasets = []
    batch_number = 0
    for batch in load_jsonl_data_in_batches(filepath, batch_size):
        batch_number += 1
        print(f"Processing batch {batch_number}...")
        df = convert_to_dataframe(batch)
        hf_data = prepare_data_for_hf(df)
        hf_dataset = convert_to_hf_dataset(hf_data)
        datasets.append(hf_dataset)
        print(f"Batch {batch_number} processed.")

    final_dataset = concatenate_datasets(datasets)
    upload_to_hf_hub(final_dataset, repo_name)
    print(f"All data uploaded to Hugging Face Hub with the repository name: {repo_name}")


principles_list = ["honesty", "prudence", "compassion", "justice", "humility", "respect"]
revision(principles_list)
convert_and_upload_to_hf_hub("revised_datafile_redteam.jsonl", "srushtisingh/Ethical")
