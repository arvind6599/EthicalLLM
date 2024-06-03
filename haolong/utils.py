import json, re, os
import random
from tqdm import tqdm
from datasets import load_dataset

def fix_common_json_errors(json_str):
    """Attempts to fix common JSON formatting errors."""
    # Remove trailing commas in objects or arrays
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str


def find_nearest_option(model_generation):
    """Find the first capital letter in the model output, and return it as the answer.
    It's not so reliable, but it's a good fallback if the high priority search fails.
    Failure cases (example):
    "The answer is D" -> will return T
    """
    len_options = 2
    best_index = len(model_generation)
    best_char = None
    characters = [chr(ord('A') + i) for i in range(len_options)] # A, B
    for char in characters:
        index = model_generation.find(char)
        if index != -1 and index < best_index:
            best_index = index
            best_char = char
    # match (a) patterns
    if best_char is None:
        for char in characters:
            pattern = f"({char.lower()})"
            if pattern in model_generation:
                best_char = char
                break
    if best_char is None:
        # everything is failing!!!!! do a random guess so that we don't raise an error, which will cause the program to ask again.
        return chr(ord('A') + random.randint(0, len_options - 1))
    return best_char

def retrieve_and_parse_response(data: str):
    """Find and parse the JSON data from the model response that may contain nested structures."""
    start_index = data.index('{')
    try:
        end_index = data.rindex('}') + 1 
        clean_data = data[start_index:end_index]
        # Attempt to parse the cleaned data
        try:
            json_data = json.loads(clean_data)
            return json_data
        except json.JSONDecodeError:
            # If the initial parse fails, try to find the correct JSON by removing nested structures
            clean_data = fix_common_json_errors(clean_data)
            pattern = r'\{[^{}]*\}'
            while True:
                matches = re.findall(pattern, clean_data)
                if not matches:
                    break  # Exit if no more brackets to evaluate
                for match in matches:
                    try:
                        json_data = json.loads(match)
                        return json_data  # Return the first successful parse
                    except json.JSONDecodeError:
                        continue  # Try the next match if the current one fails
                # Reduce the search area by removing the innermost layer of brackets
                clean_data = re.sub(pattern, '', clean_data, count=1)
            
            # raise ValueError("No JSON parsable data found in the model response after all attempts")
            first_option = find_nearest_option(data)
            return {"choice": first_option}
    except ValueError:
        # couldn't find the } in the string, probably because of too long response
        
        # find the pattern "choice": "A" or "choice": "B"
        match = re.search(r'"choice":\s*"([AB])"', data)
        if match:
            choice_str = match.group(1)
            return {"choice": choice_str}
        else:
            # nothing as "choice": "X", we just find the first "A" or "B" as it's usually what the
            # constitution AI perfers.
            match_simple = re.search(r'\s*"([AB])"', data)
            if match_simple:
                choice_str = match_simple.group(1)
                return {"choice": choice_str}
            else:
                # raise ValueError("Too long response, couldn't find anything")
                first_option = find_nearest_option(data)
                return {"choice": first_option}
            
            
def load_harmful_data():
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
    # Parsing & saving the harmful prompts
    harmful_prompts = []
    for i, sample in tqdm(enumerate(dataset['train'])):
        conversation = sample['chosen']
        harmful_prompt = conversation.split("\n\n")[1].split("Human: ")[-1]
        harmful_prompts.append(harmful_prompt)
    for i, sample in tqdm(enumerate(dataset['test'])):
        conversation = sample['chosen']
        harmful_prompt = conversation.split("\n\n")[1].split("Human: ")[-1]
        harmful_prompts.append(harmful_prompt)
    return harmful_prompts
    
def reconstruct_conversation(harmful_prompt: str, model_response: str):
    return f"""

Human: {harmful_prompt}

Assistant: {model_response}
"""

def add_rows_to_df(df, rows):
    for row in rows:
        df.loc[len(df)] = row