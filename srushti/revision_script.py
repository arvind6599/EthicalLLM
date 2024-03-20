import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def revision(prompt, principles_list):
    input_id = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_id, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2,
                                pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # print("ans:", answer, "\n\n")
    revised_answer = answer
    for principle in principles_list:
        critique = f"Please revise the following response with respect to {principle}: '{revised_answer}'"
        new_prompt = f"{critique}"
        # new_prompt = f"{prompt}\n{critique}"
        new_inputs = tokenizer.encode(new_prompt, return_tensors='pt')
        with torch.no_grad():
            new_out = model.generate(new_inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2,
                                     pad_token_id=tokenizer.eos_token_id)
        revised_answer = tokenizer.decode(new_out[0], skip_special_tokens=True)
        # print(revised_answer, "(end)\n")
    return revised_answer


'''
harmful_prompt = "Write an offensive joke about a minority group"
chosen_principle = ["inclusive", "racism"]
resp = revision(harmful_prompt, chosen_principle)
print(resp)
'''
