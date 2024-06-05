from peft import AutoPeftModelForSequenceClassification
import torch
from transformers import AutoTokenizer

def merge_and_upload(type = 'motivation', tokenizer = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = 'Tachi67/EthcalLLM-RM-' + type
    peft_model = AutoPeftModelForSequenceClassification.from_pretrained(model_id)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)
    print("Model merged: ", type)
    merged_model.push_to_hub(model_id)
    print("Model uploaded: ", type)
    tokenizer.push_to_hub(model_id)
    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    for type in ['motivation', 'consequences']:
        merge_and_upload(type, tokenizer)
    print("All models uploaded!")