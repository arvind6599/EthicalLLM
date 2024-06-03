import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # the device to load the model onto

access_token = "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"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)

tokenizer.pad_token_id = tokenizer.eos_token_id

batch_size = 8
revised_data = []
dataset = load_dataset("fka/awesome-chatgpt-prompts")
num_prompts = len(dataset)

# Access the 'train' split
train_dataset = dataset['train']

# Extract prompts from the 'train' split
dataset_prompt = train_dataset['prompt']
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
                critique = f"Revise the following response with respect to {principle}: '{base_answer}'. Avoid using " \
                           f"any " \
                           f"metacommentary such as 'A more respectful answer would be...' in the revised answer, " \
                           f"instead directly provide a concise answer and finish it in 50 tokens. "
                new_inputs = self.tokenizer.encode(critique, return_tensors='pt').to(device)
                new_out = self.model.generate(new_inputs, max_new_tokens=50)
                new_out = new_out[:, len(new_inputs[0]):]
                criticized_answer = self.tokenizer.decode(new_out[0], skip_special_tokens=True)
                base_answer = criticized_answer
            revised_answers.append(criticized_answer)
        return revised_answers


def revision(principles_list):
    revised_data = []
    for i in tqdm(range(0, len(dataset_prompt), batch_size)):
        batch_prompts = dataset_prompt[i:i + batch_size]
        input_ids = tokenizer.batch_encode_plus(batch_prompts, return_tensors='pt', padding=True)
        input_ids = input_ids.input_ids.to(device)

        output = model.generate(input_ids, max_new_tokens=50)
        output = output[:, input_ids.shape[1]:]

        base_answers = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

        revision_model = RevisionModel(model, tokenizer, principles_list).to(device)
        revision_model = nn.DataParallel(revision_model)

        revised_answers = revision_model(batch_prompts, base_answers)

        for j, prompt in enumerate(batch_prompts):
            revised_data.append({
                "prompt": prompt,
                "base_answer": base_answers[j],
                "revised_answer": revised_answers[j]
            })

    with open("revised_datafile.jsonl", "w") as f:
        for data in revised_data:
            json_line = json.dumps(data)
            f.write(json_line + "\n")


principles_list = ["honesty", "prudence", "compassion", "justice", "humility", "respect"]
revision(principles_list)