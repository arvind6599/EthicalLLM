from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline

# Load the fine-tuned model and tokenizer from the Hugging Face Hub
model_name = "srushtisingh/EthicalSFT"  # replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
'''
# Define a test prompt
test_prompts = [
    "How do I teach kids to meditate?",
    "How should we approach data privacy issues?",
    "Can you explain the importance of ethical considerations in technology?"
]

# Function to generate responses
def generate_response(prompt, model, tokenizer, max_length=100, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        num_beams=5,
        early_stopping=True
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Generate and print responses for test prompts
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    responses = generate_response(prompt, model, tokenizer)
    for i, response in enumerate(responses):
        print(f"Response {i + 1}: {response}")
    print("\n")
'''
# Use a pipeline as a high-level helper

messages = [
    {"role": "user", "content": "How do I teach kids to meditate?"},
    {"role": "user", "content": "Can you explain the importance of ethical considerations in technology?"},
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="srushtisingh/Ethicalsft")
pipe(messages)
