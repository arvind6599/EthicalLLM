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

batch_size = 16
revised_data = []
dataset = load_dataset("fka/awesome-chatgpt-prompts")[:50]
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
                critique = (f"Criticize the following response for the given prompt in terms of {principle} principle and revise it: 'prompt:{prompt}\nbase answer:{base_answer}'. "
                            f"Provide a concise, coherent and complete answer within approximately 50 tokens (or less if you can), ensuring it is a "
                            f"complete and meaningful response."
                            f"Remove additional commentary or explanations from the revised answer such as 'A more respectful answer would be...', or 'Revised response with"
                            f" respect to...' and provide only the terminal output in the revised answer."
                            f" Remove the count of tokens in the revised answer such as '(50 tokens)'."
                            f"For example:\n"
                            f"prompt: I want you to act as a linux terminal. I will type commands and you will reply "
                            f"with what the terminal should show. I want you to only reply with the terminal output "
                            f"inside one unique code block, and nothing else. do not write explanations. do not type "
                            f"commands unless I instruct you to do so. when i need to tell you something in english, "
                            f"i will do so by putting text inside curly brackets. my first command is pwd.\n"
                            f"base_answer: ```bash\n/home/user\n```"
                            f"The revised answer should be something like: '/home/username'"
                            f"What you should avoid doing for the revised answer: '\n\nRevised response:\n\n The given prompt requires the AI to "
                            f"act as a Linux terminal and only respond with the terminal output when instructed to do "
                            f"so. The base answer provided by the AI is a valid response, but it could be more "
                            f"respectful by simply providing (50 tokens)'."
                            )
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
                "Base answer": base_answers[j],
                "Revised answer": revised_answers[j]
            })

    with open("revised_datafile.jsonl", "w") as f:
        for data in revised_data:
            json_line = json.dumps(data)
            f.write(json_line + "\n")


principles_list = ["honesty", "prudence", "compassion", "justice", "humility", "respect"]
revision(principles_list)
