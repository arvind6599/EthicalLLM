import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
import time
import json
import random
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device to load the model onto

# specify how to quantize the model
# quantization_config = BitsAndBytesConfig(
#       load_in_4bit=True,
#      bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the Hugging Face dataset
with open("/home/srsingh/scratchsrsingh/revised_datafile.jsongit status",
          "r") as json_file:
    dataset = json.load(json_file)
random.shuffle(dataset)

# Split the dataset into training and evaluation sets (80% training, 20% evaluation)
train_data, eval_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Label the data
for example in train_data:
    example["split"] = "train"

for example in eval_data:
    example["split"] = "eval"

with open("train_dataset.json", "w") as train_file:
    json.dump(train_data, train_file)

with open("eval_dataset.json", "w") as eval_file:
    json.dump(eval_data, eval_file)

# dataset = load_dataset("imdb", split="train")
# Define the input and label columns
input_column = "prompt"
label_column = "revised_answer"
tokenizer.pad_token = tokenizer.eos_token
tokenized_train = tokenizer(
    [example["prompt"] for example in train_data],
    [example["revised_answer"] for example in train_data],
    # padding="max_length",
    truncation=True,
    # max_length=512,
)
tokenized_eval = tokenizer(
    [example["prompt"] for example in eval_data],
    [example["revised_answer"] for example in eval_data],
    # padding="max_length",
    truncation=True,
    # max_length=512,
)

tokenized_dataset = {"train": tokenized_train, "eval": tokenized_eval}

# Fine-tune the Mistral 7B model
# model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=len(
# set(tokenized_dataset["train"][label_column])))

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
results = trainer.evaluate()
print(results)

