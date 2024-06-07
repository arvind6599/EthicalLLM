import torch
from tqdm import tqdm
import pandas as pd
import trl 

tqdm.pandas()

import transformers
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import os

from transformers import BitsAndBytesConfig, AutoTokenizer, HfArgumentParser, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import torch
import os
import time
from peft import (
    LoraConfig,
    TaskType,
)
import wandb
from dotenv import load_dotenv

from huggingface_hub import login


load_dotenv() 


if __name__ == "__main__":

    # -------------- Initial SETUP -------------------------------------
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    login(token=HUGGINGFACE_TOKEN)
    wandb.init()

    # -------------- CONFIGURATION ------------------------------------
    
    

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # Use 4 bit quantization for the main model 
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
    )
    config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    # lora config
    LORA_R = 8
    LORA_ALPHA = 24
    LORA_DROPOUT = 0.1
    MAX_LENGTH = 300
    batch_size = 16
    DATA_SIZE = 10000
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT,
        bias='none'
    )

    ## -------------------- LOADING THE MISTRAL MODEL AND TOKENIZER -----------
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=MAX_LENGTH,
        padding="max_length", 
        truncation=True)
    
    tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        device_map=device_map,
        peft_config = lora_config
    )
    print("Model loaded!")


    ## -------------------- LOADING THE REWARD MODELS -------------------------------
    
    reward_models = []
    aspects = ['action', 'consequences', 'motivation']
    reward_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2)
    
    for aspect in aspects:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            f"Tachi67/EthcalLLM-RM-{aspect}", 
            config=reward_config,
            quantization_config=quantization_config,
            device_map=device_map
        )
        reward_model.resize_token_embeddings(len(tokenizer))
        reward_model.config.pad_token_id = reward_model.config.eos_token_id
        reward_models.append(reward_model)
    print('Reward models loaded')


    # ----------- LOADING THE PROMPTS FROM THE REWARD DATASETS --------------------- 


    def tokenize_prompt(text):
        # Find the index of "\n\nAssistant"
        index = text.find("\n\nAssistant")
        # Extract the prompt
        if index != -1:
            prompt = text[:index+len("\n\nAssistant:")]
        else:
            prompt = text  # If "\n\nAssistant" is not found, use the whole text as prompt
        return prompt

    def preprocess_function(examples, tokenizer=tokenizer):
            new_examples = {
                "query":[],
                "input_ids": [],
                "attention_mask": [],
            }
    
            for chosen in examples["chosen"]:
                prompt = tokenize_prompt(chosen)
                new_examples["query"].append(prompt)
                tokenized_prompt =  tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
                new_examples["input_ids"].append(tokenized_prompt["input_ids"])
                new_examples["attention_mask"].append(tokenized_prompt["attention_mask"])
                    
            return new_examples
    

    dataset = load_dataset("Tachi67/rm_data_action")
    train_dataset = dataset['train'].select(range(DATA_SIZE))
    test_dataset = dataset['test']
    
    prompt_train = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        )
    
    prompt_test = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        )
    
    prompt_train = prompt_train.remove_columns(["chosen", "rejected"])
    prompt_test = prompt_test.remove_columns(["chosen", "rejected"])
    
    prompt_train.set_format(type="torch")
    prompt_test.set_format(type="torch")

    # ----------------- PPO Trainer ---------------------------

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        log_with="wandb",
        batch_size=batch_size,
        mini_batch_size=1,
        gradient_accumulation_steps=batch_size,
        optimize_cuda_cache=True,
        target_kl=0.1,
        ppo_epochs=1,
        seed=42,
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # Define the PPO taii
    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=prompt_train,
        data_collator=collator
    )
    
    def tokenize_texts(texts, tokenizer):
    # Tokenize the batch of texts
        tokens = tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt')
        return tokens
    
    def get_reward_scores(texts, tokenizer, reward_model):
        # Tokenize the texts
        tokens = tokenize_texts(texts, tokenizer)
    
        # Get the input IDs and attention mask
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
    
        reward_scores = []
    
        # Get model predictions
        model.eval()
        with torch.no_grad():
            outputs = reward_model(input_ids, attention_mask=attention_mask)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Append scores for the batch (assuming binary classification and we are interested in the probability of class 1)
        batch_scores = probs[:, 1]
        reward_scores.append(batch_scores)
        
        # # Transpose the list of scores so that each inner list corresponds to the scores of one text
        # reward_scores_per_text = list(map(list, zip(*reward_scores)))
        
        return reward_scores


    generation_kwargs = {
        "min_length": -1, # don't ignore the EOS token (see above)
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 32, # specify how many tokens you want to generate at most
    }


    scale = 0.5 #factor to take the std into account to mitigate overoptimization
    
    start = time.time()
    print("Training...")

    checkpoint_dir = "./checkpoints"
    epochs = 1

    save_interval = 100  # Save checkpoint every 100 batches

    # Function to generate additional texts (motivation and consequence) in batches
    def generate_additional_texts_batch(model, tokenizer, texts, prompt, max_new_tokens):
        batched_inputs = [text + " " + prompt for text in texts]
        input_ids = tokenizer(batched_inputs, padding=True, truncation=True, return_tensors="pt").input_ids.to("cuda")
    
        # Generate response
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
        return generated_texts



    try:
        for epoch in tqdm(range(epochs), "epoch: "):
            for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader, desc="Batch")):
                
                        # Concatenate list of flattened tensors containing the input ids
                query_tensors = [ t.flatten() for t in batch['input_ids']]
                
                response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

                # Take only the new generated tokens
                new_token_responses = [response[len(query_tensors[i]):] for i, response in enumerate(response_tensors)]        
                # Decode the new tokens
                batch["response"] = [tokenizer.decode(new_tokens, skip_special_tokens=True) for new_tokens in response_tensors]
            
                scores = get_reward_scores(batch["response"], tokenizer, reward_models[0])
            
                # Generate motivations and consequences in batch
                motivation_prompt = "What is the motivation behind the Assistant's answer?"
                motivations = generate_additional_texts_batch(ppo_trainer.model, tokenizer, batch["response"], motivation_prompt, max_new_tokens=32)
                
                consequence_prompt = "What is the utility of the action towards promoting your virtuous character?"
                consequences = generate_additional_texts_batch(ppo_trainer.model, tokenizer, batch["response"], consequence_prompt, max_new_tokens=32)
                
                # Decode the motivations and consequences
                batch["motivation"] = motivations
                batch["consequence"] = consequences
                
                motivation_scores = get_reward_scores(motivations, tokenizer, reward_models[1])
                consequence_scores = get_reward_scores(consequences, tokenizer, reward_models[2])
                
                
                final_scores = torch.stack((scores[0], motivation_scores[0], consequence_scores[0])).T
                rewards = [s.mean() - scale*(s.std()) for s in final_scores]
            
                stats = ppo_trainer.step(query_tensors, new_token_responses, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)
                
                # Save checkpoint periodically
                if batch_idx % save_interval == 0:
                    checkpoint_path = f"{checkpoint_dir}/saved/"
                    ppo_trainer.model.save_pretrained(checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Saving the model before exiting...")
        ppo_trainer.model.save_pretrained(f"{checkpoint_dir}/error/")
        print("Model saved.")

                
    print("Completed!")
    print("Time taken for rm training:", time.time() - start)
    print("Pushing to hub...")
    ppo_trainer.model.push_to_hub("amenon/EthcalLLM-PPO-final")
    print("Done!")






    

    

