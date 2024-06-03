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


# def retrieve_pipe_response(pipe, prompt, max_new_tokens=32, type_pipe="SFT"):
#     """
#     Given a prompt, return the response from the pipeline
#     :param pipe: the pipeline to use
#     :param prompt: the prompt to use to the LM
#     :param max_new_tokens: the maximum number of tokens to generate
#     :return 
#         - if type_pipe == "SFT", return_dict = {"action": action, "motivation": motivation, "consequence": consequence}
#         - if type_pipe == "Constitution", returns the JSON parsable data from the model response
#     """
#     # If type is SFT, for the harmful prompt input, we ask action & motivation & consequence
#     # We ask the 3 questions sequentially by appending the chat history & user query
#     if type_pipe == "SFT":
#         return_dict = {}
#         # retriving action
#         messages = [
#             {
#                 "role":"user",
#                 "content":prompt
#             }
#         ]
#         answer_from_agent = pipe(messages, max_new_tokens=max_new_tokens, do_sample=True)[0]['generated_text'][-1]
#         return_dict["action"] = answer_from_agent['content']
        
#         # retriving motivation
#         messages.append(answer_from_agent)
#         messages.append({
#             "role":"user",
#             "content":"What is the motivation behind the answer?"
#         })
#         motivation_from_agent = pipe(messages, max_new_tokens=max_new_tokens, do_sample=True)[0]['generated_text'][-1]
#         return_dict["motivation"] = motivation_from_agent['content']
        
#         # retriving concequence
#         messages.append(motivation_from_agent)
#         messages.append({
#             "role":"user",
#             "content":"What is the utility of the action towards promoting your virtuous character?"
#         })
#         consequence_from_agent = pipe(messages, max_new_tokens=max_new_tokens, do_sample=True)[0]['generated_text'][-1]
#         return_dict["consequence"] = consequence_from_agent['content']
#         return return_dict
    
#     # If type is constitution, we ask the constitution model of the prompts generated from generate_prompt_to_constitution (see below)
#     # Most of the work here relates to paring the output from the model (see below, also see utils.py/retrieve_and_parse_response)
#     elif type_pipe == "Constitution":
#         return_dict = {}
#         messages = [
#             {
#                 "role":"user",
#                 "content":prompt
#             }
#         ]
#         # If output is cannot be handled by the parser, we ask the LM again for a valid response
#         # We don't keep asking in case the model simply fails to do so and stop.
#         # TODO: for verbosity, maybe consider skipping the current instance if things dont work out now.
#         cnt = 0
#         while True:
#             answer_from_agent = pipe(messages, max_new_tokens=max_new_tokens, do_sample=True)[0]['generated_text'][-1]
#             answer_content = answer_from_agent['content']
#             cnt += 1
#             if cnt > 3:
#                 print("********")
#                 print(answer_content)
#                 print("********")
#                 raise IndexError("Too many iterations when trying to parse")
#             try:
#                 return_dict = retrieve_and_parse_response(answer_content)
#                 return return_dict
#             except ValueError:
#                 messages.append(answer_from_agent)
#                 messages.append({
#                     "role":"user",
#                     "content":f"Previous respond is not in JSON format, please respond in JSON format within {max_new_tokens} tokens"
#                 })
#                 continue

# batch processing version of the previous function
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
        actions = pipe(initial_messages, max_new_tokens=max_new_tokens, do_sample=True)
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
        motivations = pipe(motivation_messages, max_new_tokens=max_new_tokens, do_sample=True)
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
        consequences = pipe(consequence_messages, max_new_tokens=max_new_tokens, do_sample=True)
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
        
        batch_responses = pipe(initial_messages, max_new_tokens=max_new_tokens, do_sample=True)
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

# def prepare_new_row_for_amc_data(harmful_prompt:str, response_1: str, response_2: str, virtues, type='action'):
#     assert type in ['action', 'motivation', 'consequence'], "type must be one of action, motivation, consequence"
#     new_row_1 = pd.Series({'conversation': reconstruct_conversation(harmful_prompt, response_1), 'scores': 0})
#     new_row_2 = pd.Series({'conversation': reconstruct_conversation(harmful_prompt, response_2), 'scores': 0})
#     for virtue in virtues:
#         prompt_to_constitution = generate_prompt_to_constitution(virtue, harmful_prompt, [response_1, response_2], type=type)
#         constitution_model_response = retrieve_pipe_response(pipe_constitution, prompt_to_constitution, max_new_tokens=128, type_pipe="Constitution")
        
#         if 'choice' in constitution_model_response:
#             if constitution_model_response['choice'] == 'A':
#                 new_row_1['scores'] += 1
#             elif constitution_model_response['choice'] == 'B':
#                 new_row_2['scores'] += 1
#         else:
#             response_dict_values = list(constitution_model_response.values())
#             if 'A' in response_dict_values or 'a' in response_dict_values:
#                 new_row_1['scores'] += 1
#             elif 'B' in response_dict_values or 'b' in response_dict_values:
#                 new_row_2['scores'] += 1
#             else:
#                 print(constitution_model_response)
#                 raise KeyError("wtf??")
            
#     new_row_1['scores'] = new_row_1['scores'] / len(virtues)
#     new_row_2['scores'] = new_row_2['scores'] / len(virtues)
#     return new_row_1, new_row_2

# batch processing version of the previous function
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
    constitution_responses = retrieve_pipe_responses(pipe_constitution, all_prompts, max_new_tokens=20, type_pipe="Constitution")

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
    
    model_sl.generation_config.pad_token_id = tokenizer.pad_token_id
    model_cons.generation_config.pad_token_id = tokenizer.pad_token_id

    pipe_SL = pipeline("text-generation", model=model_sl, device_map="auto", tokenizer=tokenizer
                    )
    pipe_constitution = pipeline("text-generation", model=model_cons, device_map="auto", tokenizer=tokenizer
                                )
    
    print("Pipelines loaded")
    print("SL device:", pipe_SL.device)
    print("Constitution device:", pipe_constitution.device)
    
    # suppress warning
    pipe_SL.call_count = 0
    pipe_constitution.call_count = 0
    
    
    harmful_prompts = load_harmful_data()

    virtues = ["honesty",
            "prudence",
            "justice",
            "compassion",
            "humility",
            "respect",
                ]
    
    # sampling for small scale testing
    harmful_prompts = random.choices(harmful_prompts, k=10)
    
    action_data = pd.DataFrame(columns=['conversation', 'scores'])
    motivation_data = pd.DataFrame(columns=['conversation', 'scores'])
    consequences_data = pd.DataFrame(columns=['conversation', 'scores'])
    

    # NOTE: DEPRECATED, slow
    # for harmful_prompt in tqdm(harmful_prompts, desc="Processing prompts"):
    #     SFT_model_response_dict_1 = retrieve_pipe_responses(pipe_SL, harmful_prompt, max_new_tokens=32)
    #     SFT_model_response_dict_2 = retrieve_pipe_responses(pipe_SL, harmful_prompt, max_new_tokens=32)
        
    #     new_action_data_row_1, new_action_data_row_2 = prepare_new_row_for_amc_data(harmful_prompt,
    #                                                                                 SFT_model_response_dict_1["action"],
    #                                                                                 SFT_model_response_dict_2["action"],
    #                                                                                 type='action',
    #                                                                                 virtues=virtues)
    #     new_motivation_data_row_1, new_motivation_data_row_2 = prepare_new_row_for_amc_data(harmful_prompt,
    #                                                                                         SFT_model_response_dict_1["motivation"],
    #                                                                                         SFT_model_response_dict_2["motivation"],
    #                                                                                         type='motivation',
    #                                                                                         virtues=virtues)
    #     new_consequence_data_row_1, new_consequence_data_row_2 = prepare_new_row_for_amc_data(harmful_prompt,
    #                                                                                         SFT_model_response_dict_1["consequence"],
    #                                                                                         SFT_model_response_dict_2["consequence"],
    #                                                                                         type='consequence',
    #                                                                                         virtues=virtues)
        
    #     add_rows_to_df(action_data, [new_action_data_row_1, new_action_data_row_2])
    #     add_rows_to_df(motivation_data, [new_motivation_data_row_1, new_motivation_data_row_2])
    #     add_rows_to_df(consequences_data, [new_consequence_data_row_1, new_consequence_data_row_2])
    
    start_time = time.time()
    
    # ask twice
    sft_responses_1 = retrieve_pipe_responses(pipe_SL, harmful_prompts, max_new_tokens=32, type_pipe="SFT")
    
    ########################################
    cur_time = time.time()
    print(f"Time elapsed for SFT 1: {cur_time - start_time}")
    ########################################
    
    sft_responses_2 = retrieve_pipe_responses(pipe_SL, harmful_prompts, max_new_tokens=32, type_pipe="SFT")
    
    ########################################
    cur_time = time.time()
    print(f"Time elapsed for SFT 2: {time.time() - cur_time}")
    ########################################


    action_data_rows_1, action_data_rows_2 = prepare_new_rows_for_amc_data(harmful_prompts, sft_responses_1, sft_responses_2, virtues, type='action')
    
    ########################################
    cur_time = time.time()
    print(f"Time elapsed for constituion action: {cur_time - start_time}")
    ########################################
    
    
    
    motivation_data_rows_1, motivation_data_rows_2 = prepare_new_rows_for_amc_data(harmful_prompts, sft_responses_1, sft_responses_2, virtues, type='motivation')
    
    ########################################
    cur_time = time.time()
    print(f"Time elapsed for constituion motivation: {cur_time - start_time}")
    ########################################
    
    
    
    consequence_data_rows_1, consequence_data_rows_2 = prepare_new_rows_for_amc_data(harmful_prompts, sft_responses_1, sft_responses_2, virtues, type='consequence')

    ########################################
    cur_time = time.time()
    print(f"Time elapsed for constituion consequence: {cur_time - start_time}")
    ########################################




    action_data = mix_rows(action_data_rows_1, action_data_rows_2)
    motivation_data = mix_rows(motivation_data_rows_1, motivation_data_rows_2)
    consequences_data = mix_rows(consequence_data_rows_1, consequence_data_rows_2)
    
    
    print("DataFrame DONE, converting to hf dataset")
    
    hf_action_dataset = create_hf_dataset(action_data)
    hf_motivation_dataset = create_hf_dataset(motivation_data)
    hf_consequences_dataset = create_hf_dataset(consequences_data)
    
    print("conversion DONE, pushing to hub")
    
    hf_action_dataset.push_to_hub("Tachi67/rm_data_action")
    hf_motivation_dataset.push_to_hub("Tachi67/rm_data_motivation")
    hf_consequences_dataset.push_to_hub("Tachi67/rm_data_consequences")
    
    print("push DONE")