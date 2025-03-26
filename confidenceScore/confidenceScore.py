# important, available at this point are: tokenizer, model, device, ...
# context, question, response on that question from the model.

# 1. make the yes/no prompt
prompt = make_yes_no_prompt(context, question, response)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
input_length = input_ids.shape[1]

# 2. generate the yes/no answer
#    be sure to generate output with options: output_logits=True, 
#    and return_dict_in_generate=True
outputs = model.generate(input_ids, output_logits=True, return_dict_in_generate=True, max_new_tokens=5)

# 3. calculate the yes-score 
yes_score = yes_score_calculation(outputs, input_length, tokenizer)



from typing import Any


def make_yes_no_prompt(context: str, question: str, response: str) -> str:
    return f"""Context: {context}

Question: {question}

Response: {response}

Based on the given Context and Question, answer this question:

Is the provided Response correct? Answer only Yes or No.

Answer:
    """


def yes_score_calculation(outputs: Any, input_length: int, tokenizer: Any) -> float:
    generated_tokens = outputs.sequences[:, input_length:]

    # 1. find the index (idx) of the first character-based token.
    for idx, tok in enumerate(generated_tokens[0]):
        next_token_str = tokenizer.decode(tok, skip_special_tokens=True)
        n_letters = sum(c.isalpha() for c in next_token_str)
        if n_letters != len(next_token_str):
            continue
        break
    
    # 2a. do preselection on high probabilities (out of 32k tokens)
    probs_all = torch.nn.functional.softmax(outputs.logits[idx][0], dim=-1)
    indices = torch.argwhere(probs_all > 0.001)
    indices = indices[:, -1]
    tokens_max = tokenizer.batch_decode(indices, skip_special_tokens=True)
    probs_max = probs_all[probs_all > 0.001]
    
    # 2b. find yes/no probabilities
    next_token_dict = {str(t): p for t, p in zip(tokens_max, probs_max)}
    yes_prob = next_token_dict.get("Yes", 0.)
    no_prob = next_token_dict.get("No", 0.)
    
    # 3. calculate and return yes/no confidence score
    yes_score = yes_prob / (yes_prob + no_prob) if yes_prob != 0 or no_prob != 0 else 0.5
    return yes_score