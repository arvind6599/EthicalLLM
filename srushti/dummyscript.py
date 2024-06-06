import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import json
import random
import re
import ujson


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            line = re.sub(r'\\n', '', line)
            line = re.sub(r'\\u2019', r'\\u0027', line)
            try:
                example = ujson.loads(line)
                data.append(example)
            except json.JSONDecodeError as e:
                print(f"Error: {line.strip()}")
                raise e
    return data


batch_size = 32
revised_data = []
shuffle = True

file_path = "dataset/red_team_attempts.jsonl"
sampleNumber = 32
ds = load_data(file_path)
prompts = [example['transcript'].split('\n\nHuman:')[1].split('\n\n')[0] for example in ds]
# dataset = JSONLDataset(file_path, num_samples=sampleNumber, shuffle=shuffle)
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto

access_token = "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"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)

tokenizer.pad_token_id = tokenizer.eos_token_id

# dataset = load_dataset("HuggingFaceH4/hh-rlhf")
# num_prompts = len(dataset)

# Access the 'train' split
# train_dataset = dataset['train']

# Extract prompts from the 'train' split
# dataset_prompt = train_dataset['human_turn0']
model.to(device)


# dataset_prompt = dataset_prompt[:4000]


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
    revision_model = nn.DataParallel(revision_model)

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        input_ids = tokenizer.batch_encode_plus(batch_prompts, return_tensors='pt', padding=True)
        input_ids = input_ids.input_ids.to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50)
        output = output[:, input_ids.shape[1]:]

        base_answers = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

        revised_answers = revision_model(batch_prompts, base_answers)

        for j, prompt in enumerate(batch_prompts):
            revised_data.append({
                "prompt": prompt,
                "base_answer": base_answers[j],
                "revised_answer": revised_answers[j]
            })

        torch.cuda.empty_cache()

    with open("revised_datafile_red.jsonl", "w") as f:
        for data in revised_data:
            json_line = json.dumps(data)
            f.write(json_line + "\n")


principles_list = ["honesty", "prudence", "compassion", "justice", "humility", "respect"]
revision(principles_list)
