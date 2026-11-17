# Method Comparison: Target Paper vs. This Project

Target paper: Bullwinkel et al. (2025), arXiv:2507.02956. Every choice below is either copied from the paper or logged as a deliberate deviation in `DECISIONS.md` (a local, Git-ignored notes file).

| Item | Bullwinkel et al. (2025) | This project | Note |
|---|---|---|---|
| Models | Llama-3-8B-Instruct, and Llama-3-8B-Instruct-RR (circuit breakers) | Qwen2.5-3B-Instruct (primary), OLMo-2-1B-Instruct, SmolLM2-1.7B-Instruct | No circuit-breaker variants here. The original Llama result is the reference. |
| Layer | 31 (original), 20 (circuit-breaker model) | Chosen on single-turn validation, plus one alternate layer | Never use multi-turn data to pick a layer. |
| Token position | All tokens of the final response r_n | Same | Teacher-forced final response, identical tokens across k. |
| Context variable | k = last k prompt-response pairs, k = 1 to n | Same | |
| Single-turn benign data | Retain set, 2,400 prompt-response pairs | 100 records sampled from the retain set, redacted | Not topic-matched to the harmful set, so style differences may help the probe. |
| Single-turn harmful data | Circuit-breaker set, 2,400 prompt-response pairs | 100 records sampled from the circuit-breaker set, redacted | Not restricted to the two categories. JailbreakBench and HarmBench were not used. |
| Representation | 4096-d token vectors (Llama-3-8B) | Model hidden size | Set per model in `configs/models.yaml`. |
| Train/test split | Random 80/20 over pooled tokens | Grouped by example id | Removes within-response leakage. |
| PCA | Fitted on training tokens, first 2 components, 6.7% variance (original model) | Same, plus variance explained at more components, and centroid distances in the top components | A 2-d plot alone shows little for this effect. |
| Probe | scikit-learn MLP, hidden (64, 32), max_iter 1000, random_state 42 | Same architecture, plus a linear probe | Same settings keep results comparable. |
| Probe sanity check | Test accuracy 0.997 and 0.999 | Grouped held-out examples: token accuracy 94.2% (Qwen), 92.1% (OLMo), 93.6% (SmolLM2); example accuracy 96.7%, 96.7%, 100% | Lower than the paper, as expected from the smaller set and the example-grouped split. |
| Multi-turn data | 5 manual Crescendo attacks in depth (molotov, selfharm, meth, phishing, firearm), 4 to 7 turns | Round one: 4 + 4 hand-written conversations with refusal replies. Round two: 50 + 50 self-authored user-turn sets per model, each with that model's own greedy replies | Round two prompts are templated arcs, so diversity is limited, and turns 1 and 2 are identical within each escalation and control pair. Report per conversation and with confidence intervals. |
| Headline metric | Percentage of final-response tokens classified harmful, per k | Same, plus a conversation-level decision and the earliest k flagged | |
| RQ3 conditions | Crescendo, Single Prompt, Masked Responses, Attack Objective | Crescendo, Single Prompt, Attack Objective. Masked Responses not implemented | Attack Objective is the positive control. |
| Benign multi-turn control | None | Matched benign conversations | Tests whether the k = 1 to 2 drop is specific to escalation. |
| Attack success check | PyRIT scorer plus manual verification against written criteria | Manual refused / partial / complied label on each final reply. Complied: 14 of 50 (Qwen), 26 (OLMo), 47 (SmolLM2). No written rubric for partial yet; the results notebook repeats the key tests on complied replies only | An audit of the Qwen labels found partial used for two different behaviours. Relabelling is pending. |
| Black-box baseline | None | Frozen prompt, per-turn judge with a second review pass. Round one used local llama3.2:3b (32 rows). Round two uses Ministral-3-8B-Instruct (GGUF Q4_K_M) through Ollama on a Colab GPU, 400 scored turns per model, prompt hash checked at run time | New in this project. The `judge` block of `configs/experiment.yaml` still names a Groq endpoint; the model recorded in each result row is the source of truth. |
| Reply generation | Replies from the attacked model during the attack | Greedy decoding, capped at 160 new tokens, 4-bit weights | Most final replies end mid-sentence, so "complied" means "began to comply". |
| Defense context | Circuit breakers | None | |
