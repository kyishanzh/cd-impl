import transformers as tr
import torch

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

    return input_ids


if __name__ == "__main__":
    amateur_model, expert_model = load_models()
    print(contrastive_generation(amateur_model, expert_model, prompt, 128))
