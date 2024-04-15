import gzip

from datasets import load_dataset
import json


def extractFile():
    # Load the dataset
    dataset = load_dataset("fka/awesome-chatgpt-prompts")

    # Access the 'train' split
    train_dataset = dataset['train']

    # Extract prompts from the 'train' split
    prompts = train_dataset['prompt']

    # Create a list to store prompts as JSON objects
    prompts_json = [{'prompt': prompt} for prompt in prompts]

    # Save prompts to a JSON file
    with open('chatgpt_prompts.jsonl', 'w', encoding='utf-8') as f:
        json.dump(prompts_json, f, ensure_ascii=False, indent=4)
        f.write('\n')


def extractFileJsonL(input_file, output_file):
    with open(input_file, encoding='utf-8') as f_in, open(output_file, 'w') as f_out:

        convo = json.load(f_in)
        for l in convo:
            transcript = l.get("transcript")
            if transcript:
                lines = transcript.split('\n\n')
                if len(lines) >= 2:
                    for line in lines:
                        if line.startswith("Human:"):
                            human_prompt = line.split(':')[1].strip()
                            f_out.write(json.dumps({"prompt": human_prompt}))
                            f_out.write('\n')
                            break


input = 'C:\\Users\\Tanu.DESKTOP-D6I4PBO\\Documents\\epfl\\bachelor project\\red_team_attempts.jsonl'
output = 'red_team_attempts_dataset.jsonl'
extractFileJsonL(input, output)
