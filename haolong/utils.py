import json, re, os

def fix_common_json_errors(json_str):
    """Attempts to fix common JSON formatting errors."""
    # Remove trailing commas in objects or arrays
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str

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
            
            raise ValueError("No JSON parsable data found in the model response after all attempts")
    except ValueError:
        # couldn't find the } in the string, probably because of too long response
        match = re.search(r'"choice":\s*"([AB])"', data)
        if match:
            choice_str = match.group(1)
            return {"choice": choice_str}
        else:
            match_simple = re.search(r'\s*"([AB])"', data)
            if match_simple:
                choice_str = match_simple.group(1)
                return {"choice": choice_str}
            else:
                raise ValueError("Too long response, couldn't find anything, see input above")
            
            
def load_harmful_data():
    if os.path.exists('/scratch/students/haolli/harmful_prompts_anthropic_hh.json'):
        # load the data directly if exists
        with open('/scratch/students/haolli/harmful_prompts_anthropic_hh.json', 'r') as file:
            harmless_prompts = json.load(file)
        return harmless_prompts
    else:
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", cache_dir="/scratch/students/haolli")
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

        with open('/scratch/students/haolli/harmful_prompts_anthropic_hh.json', 'w') as file:
            json.dump(harmful_prompts, file)
        return harmful_prompts