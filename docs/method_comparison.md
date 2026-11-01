# Method Comparison: Target Paper vs. This Project

Target paper: Bullwinkel et al. (2025), arXiv:2507.02956. Every choice below is either copied from the paper or logged as a deliberate deviation in `DECISIONS.md`.

| Item | Bullwinkel et al. (2025) | This project | Note |
|---|---|---|---|
| Models | Llama-3-8B-Instruct, and Llama-3-8B-Instruct-RR (circuit breakers) | Qwen2.5-3B-Instruct (primary), OLMo-2-1B-Instruct, Moonlight-16B-A3B-Instruct | No circuit-breaker variants here. The original Llama result is the reference. |
| Layer | 31 (original), 20 (circuit-breaker model) | Chosen on single-turn validation, plus one alternate layer | Never use multi-turn data to pick a layer. |
| Token position | All tokens of the final response r_n | Same | Teacher-forced final response, identical tokens across k. |
| Context variable | k = last k prompt-response pairs, k = 1 to n | Same | |
| Single-turn benign data | Retain set, 2,400 prompt-response pairs | Matched benign prompts, same topic, length and turn count | Redacted or self-authored. |
| Single-turn harmful data | Circuit-breaker set, 2,400 prompt-response pairs | JailbreakBench or HarmBench prompts in two categories, redacted | Phishing/fraud and non-operational cyber intrusion. |
| Representation | 4096-d token vectors (Llama-3-8B) | Model hidden size | Set per model in `configs/models.yaml`. |
| Train/test split | Random 80/20 over pooled tokens | Grouped by example id | Removes within-response leakage. |
| PCA | Fitted on training tokens, first 2 components, 6.7% variance (original model) | Same, plus variance explained at more components, and centroid distances in the top components | A 2-d plot alone shows little for this effect. |
| Probe | scikit-learn MLP, hidden (64, 32), max_iter 1000, random_state 42 | Same architecture, plus a linear probe | Same settings keep results comparable. |
| Probe sanity check | Test accuracy 0.997 and 0.999 | Report on grouped held-out examples | Expect a lower number than the paper. |
| Multi-turn data | 5 manual Crescendo attacks in depth (molotov, selfharm, meth, phishing, firearm), 4 to 7 turns | Fixed prewritten conversations, two categories | Small n: report per conversation. |
| Headline metric | Percentage of final-response tokens classified harmful, per k | Same, plus a conversation-level decision and the earliest k flagged | |
| RQ3 conditions | Crescendo, Single Prompt, Masked Responses, Attack Objective | Crescendo, Single Prompt, Attack Objective. Masked Responses if time allows | Attack Objective is the positive control. |
| Benign multi-turn control | None | Matched benign conversations | Tests whether the k = 1 to 2 drop is specific to escalation. |
| Attack success check | PyRIT scorer plus manual verification against written criteria | Written criteria, manual check, contingency if no unsafe output | |
| Black-box baseline | None | Frozen LLM judge scoring escalation per turn | New in this project. |
| Defense context | Circuit breakers | None | |

