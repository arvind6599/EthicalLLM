from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer

write_token = "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"
access_token = "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"

def merge_and_upload(tokenizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = 'srushtisingh/EthicalSFT'
    peft_model = AutoPeftModelForCausalLM.from_pretrained(model_id)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)
    print("Model merged: ")
    merged_model.push_to_hub(model_id, token=write_token)
    print("Model uploaded: ")
    tokenizer.push_to_hub(model_id, token=write_token)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=access_token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    merge_and_upload(tokenizer)
    print("All models uploaded!")
