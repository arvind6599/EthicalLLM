from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json
import numpy as np
from typing import List
from datasets import Dataset
import pandas as pd
import random
import time
from utils import retrieve_and_parse_response, load_harmful_data, reconstruct_conversation, add_rows_to_df

def retrieve_pipe_responses(pipe, prompts, max_new_tokens=32, type_pipe="SFT"):
    """
    Given a list of prompts, return the responses from the pipeline in batch
    :param pipe: the pipeline to use
    :param prompts: a list of prompts to use for the LM
    :param max_new_tokens: the maximum number of tokens to generate
    :return a list of results according to the type_pipe
    """
    responses = []

    if type_pipe == "SFT":
        # Initialize the messages for each prompt
        initial_messages = [
            [{
                "role": "user",
                "content": prompt
            }]
            for prompt in prompts
        ]

        # Batch processing for actions
        actions = pipe(initial_messages, max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=2, pad_token_id=2)
        actions = [action[0]['generated_text'][-1] for action in actions]

        # Update messages with actions and new query for motivations
        motivation_messages = [
            messages + [action, {
                "role": "user",
                "content": "What is the motivation behind the answer?"
            }]
            for messages, action in zip(initial_messages, actions)
        ]

        # Batch processing for motivations
        motivations = pipe(motivation_messages, max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=2, pad_token_id=2)
        motivations = [motivation[0]['generated_text'][-1] for motivation in motivations]

        # Update messages with motivations and new query for consequences
        consequence_messages = [
            messages + [motivation, {
                "role": "user",
                "content": "What is the utility of the action towards promoting your virtuous character?"
            }]
            for messages, motivation in zip(motivation_messages, motivations)
        ]

        # Batch processing for consequences
        consequences = pipe(consequence_messages, max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=2, pad_token_id=2)
        consequences = [consequence[0]['generated_text'][-1] for consequence in consequences]

        # Compose final responses
        responses = [
            {
                "action": action['content'],
                "motivation": motivation['content'],
                "consequence": consequence['content']
            }
            for action, motivation, consequence in zip(actions, motivations, consequences)
        ]

    elif type_pipe == "Constitution":
        # Initialize messages for each prompt
        initial_messages = [
            [{
                "role": "user",
                "content": prompt
            }]
            for prompt in prompts
        ]
        
        batch_responses = pipe(initial_messages, max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=2, pad_token_id=2)
        batch_responses = [response[0]['generated_text'][-1] for response in batch_responses]
        responses = batch_responses


    return responses

def generate_prompt_to_constitution(virtue: str, harmful_prompt_to_SFT_model: str, SFT_model_response: List[str], type='action') -> str:
    """
    Generate the prompt for the constitution model.
    :param virtue: the virtue to be considered, it's one of the elements in the virtues list
    :param harmful_prompt_to_SFT_model: the harmful prompt to the SFT model (human query)
    :param SFT_model_response: the two responses from the SFT model based on the harmful prompt
    :return prompt_to_constitution: the prompt for the constitution model
    """
    
    assert type in ['action', 'motivation', 'consequence'], "type must be one of action, motivation, consequence"
    
    prompt_to_constitution = f"""
    The following question was asked by the user:
    Human: {harmful_prompt_to_SFT_model}
    The assitant responded with the {type} regarding the question.
    Which of the following response is more {virtue} ?
    ---- Start of assistant responded {type} options ----
    Options:
    (A){SFT_model_response[0]}
    (B){SFT_model_response[1]}
    ---- End of assistant response options ----
    Your response must be purely in JSON format, with the following content:
    {{
    "choice": "A" or "B",
    }}
    """
    return prompt_to_constitution

def prepare_new_rows_for_amc_data(harmful_prompts, responses_1, responses_2, virtues, type='action'):
    new_rows_1 = []
    new_rows_2 = []
    
    all_prompts = []
    prompt_to_index_virtue_map = []
    
    # generate all prompts for the constitution model
    for index, harmful_prompt in enumerate(harmful_prompts):
        response_1 = responses_1[index][type]
        response_2 = responses_2[index][type]
        for virtue in virtues:
            prompt = generate_prompt_to_constitution(virtue, harmful_prompt, [response_1, response_2], type=type)
            all_prompts.append(prompt)
            prompt_to_index_virtue_map.append((index, virtue))
    
    # process all prompts in batch
    constitution_responses = retrieve_pipe_responses(pipe_constitution, all_prompts, max_new_tokens=16, type_pipe="Constitution")

    # initialize scores
    scores_1 = [0] * len(harmful_prompts)
    scores_2 = [0] * len(harmful_prompts)

    # handle the responses from the constitution model
    for response, (index, virtue) in zip(constitution_responses, prompt_to_index_virtue_map):
        if 'choice' in response:
            if response['choice'] == 'A':
                scores_1[index] += 1
            elif response['choice'] == 'B':
                scores_2[index] += 1

    # convert the scores to the final scores
    for index, harmful_prompt in enumerate(harmful_prompts):
        response_1 = responses_1[index][type]
        response_2 = responses_2[index][type]
        new_row_1 = pd.Series({'conversation': reconstruct_conversation(harmful_prompt, response_1), 'scores': scores_1[index] / len(virtues)})
        new_row_2 = pd.Series({'conversation': reconstruct_conversation(harmful_prompt, response_2), 'scores': scores_2[index] / len(virtues)})
        new_rows_1.append(new_row_1)
        new_rows_2.append(new_row_2)

    return new_rows_1, new_rows_2


def mix_rows(data_1, data_2):
    mixed_data_rows = []
    for row1, row2 in zip(data_1, data_2):
        mixed_data_rows.append(row1)
        mixed_data_rows.append(row2)
    mixed_data = pd.DataFrame(mixed_data_rows)
    return mixed_data

def create_hf_dataset(df):
    grouped_data = {
        "chosen": [],
        "rejected": []
    }

    for i in range(0, len(df), 2):
        if df.loc[i, "scores"] >= df.loc[i+1, "scores"]:
            grouped_data["chosen"].append(df.loc[i, "conversation"])
            grouped_data["rejected"].append(df.loc[i+1, "conversation"])
        else:
            grouped_data["chosen"].append(df.loc[i+1, "conversation"])
            grouped_data["rejected"].append(df.loc[i, "conversation"])

    # Convert the grouped data into a HuggingFace dataset
    hf_dataset = Dataset.from_dict(grouped_data)
    return hf_dataset



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # quantize for the sake of memory
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
    )

    model_sl = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=quantization_config, device_map="auto")
    model_cons = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=quantization_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    model_sl.generation_config.pad_token_id = model_sl.generation_config.eos_token_id
    model_cons.generation_config.pad_token_id = model_cons.generation_config.eos_token_id

    pipe_SL = pipeline("text-generation", model=model_sl, device_map="auto", tokenizer=tokenizer, batch_size=160)
    pipe_constitution = pipeline("text-generation", model=model_cons, device_map="auto", tokenizer=tokenizer, batch_size=160)
    
    pipe_SL.tokenizer.pad_token_id = model_sl.config.eos_token_id
    pipe_constitution.tokenizer.pad_token_id = model_cons.config.eos_token_id
    
    print("Pipelines loaded")
    print("SL device:", pipe_SL.device)
    print("Constitution device:", pipe_constitution.device)
    
    
    
    harmful_prompts = load_harmful_data()

    virtues = ["honesty",
            "prudence",
            "justice",
            "compassion",
            "humility",
            "respect",
                ]
    
    # sampling for small scale testing
    harmful_prompts = random.choices(harmful_prompts, k=200)
    
    action_data = pd.DataFrame(columns=['conversation', 'scores'])
    motivation_data = pd.DataFrame(columns=['conversation', 'scores'])
    consequences_data = pd.DataFrame(columns=['conversation', 'scores'])
    
    
    cur_time = time.time()
    start_time = cur_time
    
    # ask twice
    sft_responses_1 = retrieve_pipe_responses(pipe_SL, harmful_prompts, max_new_tokens=64, type_pipe="SFT")
    
    ########################################
    print(f"Time elapsed for SFT 1: {time.time() - cur_time}")
    cur_time = time.time()
    ########################################
    
    sft_responses_2 = retrieve_pipe_responses(pipe_SL, harmful_prompts, max_new_tokens=64, type_pipe="SFT")
    
    ########################################
    print(f"Time elapsed for SFT 2: {time.time() - cur_time}")
    cur_time = time.time()
    ########################################


    action_data_rows_1, action_data_rows_2 = prepare_new_rows_for_amc_data(harmful_prompts, sft_responses_1, sft_responses_2, virtues, type='action')
    
    ########################################
    print(f"Time elapsed for constituion action: {time.time() - cur_time}")
    cur_time = time.time()
    ########################################
    
    
    
    motivation_data_rows_1, motivation_data_rows_2 = prepare_new_rows_for_amc_data(harmful_prompts, sft_responses_1, sft_responses_2, virtues, type='motivation')
    
    ########################################
    print(f"Time elapsed for constituion motivation: {time.time() - cur_time}")
    cur_time = time.time()
    ########################################
    
    
    
    consequence_data_rows_1, consequence_data_rows_2 = prepare_new_rows_for_amc_data(harmful_prompts, sft_responses_1, sft_responses_2, virtues, type='consequence')

    ########################################
    print(f"Time elapsed for constituion consequence: {time.time() - cur_time}")
    cur_time = time.time()
    ########################################




    action_data = mix_rows(action_data_rows_1, action_data_rows_2)
    motivation_data = mix_rows(motivation_data_rows_1, motivation_data_rows_2)
    consequences_data = mix_rows(consequence_data_rows_1, consequence_data_rows_2)
    
    
    print(f"Time elapsed for all: {time.time() - start_time}")
    
    
    print("DataFrame DONE, converting to hf dataset")
    
    hf_action_dataset = create_hf_dataset(action_data)
    hf_motivation_dataset = create_hf_dataset(motivation_data)
    hf_consequences_dataset = create_hf_dataset(consequences_data)
    
    print("conversion DONE, pushing to hub")
    
    hf_action_dataset.push_to_hub("Tachi67/rm_data_action")
    hf_motivation_dataset.push_to_hub("Tachi67/rm_data_motivation")
    hf_consequences_dataset.push_to_hub("Tachi67/rm_data_consequences")
    
    print("push DONE")