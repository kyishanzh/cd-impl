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


def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
    # tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(expert.device)

    expert_past = None
    amateur_past = None

    generated_ids = input_ids

    for _ in tqdm(range(max_tokens), "Generating response with CD"):
        # run expert model
        expert_out = expert.generate(input_ids=generated_ids, use_cache=True, past_key_values=expert_past)
        expert_logits = expert_out.logits[:, -1, :]
        expert_past = expert_out.past_key_values

        # run amateur model
        amateur_out = amateur.generate(input_ids=generated_ids, use_cache=True, past_key_values=amateur_past)
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
        contrast_scores[~mask] = -999999.0  # ignore tokens that did not pass plausibility constraint
        # pick token w/ max contrast score as next token
        next_token = torch.argmax(contrast_scores, dim=-1).unsqueeze(-1)
        assert next_token.shape == [1, 1]
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # check if Qwen is done generating (eos)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    amateur_model, expert_model = load_models()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(expert_model.device)
    amateur_response = tokenizer.decode(amateur_model(input_ids, max_new_tokens=128))
    expert_response = tokenizer.decode(expert_model(input_ids, max_new_tokens=128))
    cd_response = contrastive_generation(amateur_model, expert_model, prompt, max_tokens=128)
