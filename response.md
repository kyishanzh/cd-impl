# Responses to questions :)
## Question 1
**What should you do if the two models have different tokenizers?**

**Answer:** If the two models have different tokenizers, we could potentially try to fine-tune one of the models with the tokenizer of the other model such that they are aligned for the contrastive decoding process. There also exist tokenizer “translators” (e.g. https://arxiv.org/html/2405.07883v1) that we could potentially use to translate one tokenizer to the other. 

## Question 2
**Do you think contrastive decoding is used in practice?**

**Answer:** I think it would make sense to use contrastive decoding in practice - if we know that smaller LMs tend to make more mistakes in ways that larger LMs sometimes do as well, CD provides a neat way to specify particular response attributes that we don’t want without needing us to actively dissect what we desire in an ideal response. Having AI systems with multiple agents / multiple models running at the same time is also becoming increasingly popular (e.g., having multiple experts argue with each other before providing the user with a response), and CD is one particular way we could leverage information from multiple agents to improve the output.