import transformers as tr
import torch
from tqdm import tqdm

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path, trust_remote_code=True)

ALPHA = 0.1  # constant specified in the paper

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def load_models():
    """
    Load models from HuggingFace.
    :return:
    """
    expert = tr.AutoModelForCausalLM.from_pretrained(expert_path, device_map="auto")
    amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path, device_map="auto")
    print("Models loaded!")
    return amateur, expert


def contrastive_generation(amateur, expert, prompt, max_tokens=None) -> str:
    # tokenize prompt
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded.input_ids.to(expert.device)
    attention_mask = encoded.attention_mask.to(expert.device)

    # initialize generation
    generated_ids = input_ids.clone()

    # progress bar
    progress_max = max_tokens if max_tokens is not None else 1000
    pbar = tqdm(total=progress_max, desc="Generating response with CD")

    expert_past = None
    amateur_past = None
    token_count = 0

    while True:
        # check if we've reached the maximum number of tokens
        if max_tokens is not None and token_count >= max_tokens:
            break

        # for the first token, use the full input; for subsequent tokens, only use the last token
        if generated_ids.shape[1] == input_ids.shape[1]:
            # first token generation - use full input
            curr_input_ids = generated_ids
            expert_past = None
            amateur_past = None
        else:
            # subsequent tokens - only use the last generated token
            curr_input_ids = generated_ids[:, -1].unsqueeze(-1)

        # run expert model
        expert_out = expert(
            input_ids=curr_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=expert_past
        )
        expert_logits = expert_out.logits[:, -1, :]
        expert_past = expert_out.past_key_values

        # run amateur model
        amateur_out = amateur(
            input_ids=curr_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=amateur_past
        )
        amateur_logits = amateur_out.logits[:, -1, :]
        amateur_past = amateur_out.past_key_values

        # converting to log probabilities
        expert_logprobs = torch.log_softmax(expert_logits, dim=-1)
        amateur_logprobs = torch.log_softmax(amateur_logits, dim=-1)

        # implementing plausibility constraint: log(p_exp) >= log(alpha) + log(max probability)
        log_maxprobs = torch.max(expert_logprobs, dim=-1, keepdim=True).values
        cutoff = torch.log(torch.tensor(ALPHA, device=expert_logprobs.device)) + log_maxprobs
        mask = expert_logprobs >= cutoff  # all tokens that passed plausibility constraint

        assert mask.shape[0] == 1

        # implementing CD
        contrast_scores = expert_logprobs - amateur_logprobs
        contrast_scores[~mask] = -float('inf')  # ignore tokens that did not pass plausibility constraint

        # pick token w/ max contrast score as next token
        next_token = torch.argmax(contrast_scores, dim=-1).unsqueeze(-1)

        assert next_token.shape == (1, 1)

        # add the new token to generated_ids
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # update attention mask for the new token
        new_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        # increment token counter
        token_count += 1
        pbar.update(1)

        # check if we're done generating (reached EOS token)
        if next_token.item() == tokenizer.eos_token_id:
            break

    pbar.close()

    # extract only the newly generated tokens (excluding the input prompt)
    generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

    return generated_text


def vanilla_generation(model, prompt, max_tokens) -> str:
    """
    Generate text using only the expert model with the same token-by-token approach
    as the contrastive decoding implementation for fair comparison.
    """
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    generated_ids = input_ids.clone()
    past_key_values = None

    for _ in tqdm(range(max_tokens), desc="Vanilla generation"):
        if generated_ids.shape[1] == input_ids.shape[1]:
            curr_input_ids = generated_ids
        else:
            curr_input_ids = generated_ids[:, -1].unsqueeze(-1)

        outputs = model(
            input_ids=curr_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values
        )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        new_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    amateur_model, expert_model = load_models()
    cd_response = contrastive_generation(amateur_model, expert_model, prompt, max_tokens=128)

    print("CD RESPONSE ---------\n", cd_response)

    expert_response = vanilla_generation(expert_model, prompt, 128)
    print("EXPERT RESPONSE ---------\n", expert_response)

    amateur_response = vanilla_generation(amateur_model, prompt, 128)
    print("AMATEUR RESPONSE ---------\n", amateur_response)

