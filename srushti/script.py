import json
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

with open('train.jsonl', encoding='utf-8') as datafile:
    dataset = [json.loads(l) for l in datafile]

for document in dataset:
    convo = document['chosen']
    all_utterance = re.findall(r'Human: (.*)\n', convo)
    if all_utterance:
        first_utterance = all_utterance[0]
        inputs = tokenizer(first_utterance, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)  # max_length is the max_length of the response of the model,
        # temperature is the randomness...ask how to set these parameters
        text = tokenizer.decode(outputs[0])
        print("Input:", first_utterance)
        print("Generated Response:", text)
