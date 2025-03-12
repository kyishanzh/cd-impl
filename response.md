# Responses to questions :)
## Question 1
**What should you do if the two models have different tokenizers?**

**Answer:** If the two models have different tokenizers, ...
Contrastive Decoding relies on comparing logits for identical token indices, so both models must treat your current sequence with a consistent vocabulary. If the models have incompatible tokenizers, you need to do some alignment. A simple—but sometimes imperfect—approach is to force both models to use the larger model’s tokenizer, encoding the text with that tokenizer for both models. You’d then ignore or remap any unknown tokens or mismatches when running the smaller model. Another option is to train or adapt one model to match the other’s tokenizer. These workarounds may not be perfectly accurate, but they’re necessary whenever the two models do not share an identical vocabulary.

## Question 2
**Do you think contrastive decoding is used in practice?**

**Answer:** I think it would make sense to use contrastive decoding in practice - if we know that smaller LMs tend to make more mistakes in ways that larger LMs sometimes do as well, CD provides a neat way to specify particular response attributes that we don’t want without needing us to actively dissect what we desire in an ideal response. Having AI systems with multiple agents / multiple models running at the same time is also becoming increasingly popular (e.g., having multiple experts argue with each other before providing the user with a response), and CD is one particular way we could leverage information from multiple agents to improve the output.