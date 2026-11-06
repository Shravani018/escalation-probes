# Literature Review

Grounded in three papers, each read in full. Figure, table and section numbers refer to the arXiv versions below.

| Short name | Reference | Venue |
|---|---|---|
| Crescendo | Russinovich, Salem and Eldan (2024). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. arXiv:2404.01833v3 | USENIX Security 2025 |
| Refusal direction | Arditi et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717v3 | NeurIPS 2024 |
| Target paper | Bullwinkel et al. (2025). A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks. arXiv:2507.02956v1 | DIG-BUGS workshop, ICML 2025 |

Works cited inside these papers are listed in section 6 and have not been read. Section 7 lists sources the project still needs that are not in the attached papers.

## 1. Crescendo (Russinovich, Salem and Eldan, 2024)

**Attack.** Crescendo is a black-box, multi-turn jailbreak. It opens with an abstract, benign question about the target topic and escalates over several turns by referring to the model's own earlier replies. Each user message is readable and harmless on its own (Sections 1 and 3). The authors compare the pattern to the foot-in-the-door tactic. In the automated version, a refused question is removed from the target's history and rephrased, up to ten times.

**Evidence.** Manual runs succeeded on nearly all tested tasks against ChatGPT (GPT-4), Gemini Pro and Ultra, Claude-2 and Claude-3, LLaMA-2 70b and LLaMA-3 70b (Table 2). The automated tool, Crescendomation, reached an average attack success rate (ASR) of 56.2% on GPT-4 and 82.6% on Gemini Pro across 50 AdvBench tasks. The strongest baseline reached 40.0% and 42.4% (Table 4). On 100 HarmBench tasks against GPT-4 it reached 63.2%, against 38.9% for many-shot jailbreaking (Figure 9). Most tasks were broken in fewer than five turns (Table 5).

**Behavioral account.** On LLaMA-2 70b the authors track the probability that a reply opens with "Sure" (compliance) or "I" (refusal). A mid-sequence request succeeds 36.2% of the time on its own and 99.99% after a preceding general turn. The final request succeeds 17.3% after only the second turn, 99.9% after the full sequence, and under 1% when it states the task outright (Table 3). Removing the most influential generated sentence does not stop the effect, so the authors attribute it to the accumulated context as a whole (Figure 5). This account is behavioral. It says nothing about internal state.

**Judging.** A GPT-4 judge returns a success flag, a percentage and a written rationale. A second judge reviews the rationale, because the first judge sometimes agrees the task was done yet declines to call it a success on safety grounds. A separate refusal judge triggers backtracking. Top results are checked by hand, and Perspective API and Azure Content Filter scores serve as external checks (Section 4.2).

**Size and transfer.** LLaMA-2 7b and 70b respond almost identically, so for that family vulnerability does not track size (Section 5.4, Figure 10). Questions taken from one model and replayed on another work less well than questions tailored to the target, and some tasks transfer at close to 0% (Section 5.6). The authors call replayed conversations a lower bound.

**Relevance here.**
- Small open models are plausible Crescendo targets.
- Fixed prewritten conversations match the transfer setting, so a low unsafe-output rate would reflect a weaker attack than Crescendomation. This supports keeping the contingency plan.
- The judge design is a template for the black-box baseline. It scores task completion, whereas this project needs a per-turn escalation score.

## 2. Arditi et al. (2024)

**Method.** For each layer and each post-instruction token position, a candidate direction is the difference between mean activations on harmful and harmless instructions. Harmful instructions come from AdvBench, MaliciousInstruct, TDC2023 and HarmBench (128 train, 32 validation). Harmless instructions come from Alpaca. One vector is selected using three validation scores: bypass (refusal metric under ablation), induce (refusal metric under addition) and KL divergence on harmless prompts. The constraints are induce score above 0, KL below 0.1 and layer below 0.8 of model depth (Section 2.3, Appendix C).

**Findings.** Across 13 chat models from 1.8B to 72B parameters (Qwen, Yi, Gemma, Llama-2, Llama-3), erasing the direction from the residual stream removes refusal and raises unsafe completions on JailbreakBench. Adding it induces refusal on harmless Alpaca prompts (Figures 1 and 3). Folding the ablation into the weights gives HarmBench ASRs on par with prompt-specific attacks on Qwen models, for example 79.2% for Qwen 7B (Table 2). Refusal is scored by substring match and harmfulness by Llama Guard 2 (Section 2.5). The direction is already expressed in base models, which suggests safety tuning reuses it (Appendix J).

**Adversarial suffix.** On one model (Qwen 1.8B Chat) and one suffix, appending the suffix pushes the last-token cosine similarity with the refusal direction down to the level of harmless prompts. It also moves the attention of the heads that write to the direction from the instruction to the suffix (Section 5, Figures 5 and 6). The authors note that this conflicts with Zou et al. (2023a), who found harmfulness representations largely unchanged by suffixes (Section 6).

**Stated limits.** The direction's meaning is unclear and may encode harm rather than refusal. Extraction relies on heuristics. The suffix analysis covers one model and one example (Section 7).

**Relevance here.**
- The paper supplies the single-turn direction the project asks about, and shows a single-turn jailbreak can work by suppressing it at one token position.
- It never tests multi-turn conversations. The Week 1 question, whether a single-turn refusal direction misses gradual escalation, has no direct answer here.
- It reads activations at post-instruction positions, the decision point before a reply. The target paper reads response tokens. The two readouts are not interchangeable.
- Qwen Chat models appear, but not Qwen2.5, OLMo-2 or Gemma 3.
- Tooling: TransformerLens for exploration, and HuggingFace Transformers, PyTorch and vLLM for the pipeline. Code is at github.com/andyrdt/refusal_direction.

## 3. Bullwinkel et al. (2025), the target paper

**Questions.** RQ1: how models represent Crescendo inputs compared with single-turn inputs. RQ2: how the number of conversation turns changes those representations. RQ3: how the model's own responses shape them.

**Setup.** The models are Llama-3-8B-Instruct and Llama-3-8B-Instruct-RR, which was fine-tuned with circuit breakers (Zou et al., 2024). Ten attack objectives were run through PyRIT with 20 automated Crescendo attacks each, giving ASRs of 64.2% on the original and 54.2% on the circuit-breaker model (Table 1). Zou et al. report an average of 3.8% for unseen single-turn attacks on the circuit-breaker model, which the authors read as a generalization gap. Five attacks (molotov, selfharm, meth, phishing, firearm) were performed manually, to verify they met the success criteria in Appendix B, and analysed in depth. They ran 4 to 7 turns.

**Method.** The object of study is the activation at the tokens of the final response at one layer, with the last k prompt-response pairs in context, for k = 1 to n (Equations 1 and 3). Layer 31 is used for the original model and layer 20 for the circuit-breaker model. Single-turn data are 2,400 prompt-response pairs each from a retain set (benign) and a circuit-breaker set (harmful). Each token is a 4096-dimensional vector, split at random 80/20. PCA is fitted on the training split and the first two components are plotted, explaining 6.7% of variance for the original model. The probe is a scikit-learn MLP with hidden layers (64, 32), reaching test accuracy 0.997 and 0.999 (Section 3.2, Appendix F). The headline metric is the percentage of final-response tokens classified as harmful.

**Results.**
- RQ1: For the original model, benign and harmful single-turn data separate only marginally. Final Crescendo tokens sit slightly nearer the benign cluster, while the same objective asked directly sits nearer the harmful one (Figure 1a). The effect is much clearer for the circuit-breaker model (Figure 1b).
- RQ2: The share of tokens flagged harmful falls as k grows, usually with a sharp drop from k = 1 to k = 2, except for the firearm case (Figure 2). At k = n the original model still flags roughly a third of molotov tokens and a little over a quarter of meth tokens (values read from the figure). The other three attacks are near zero.
- RQ3: Passing the history as a single user prompt gives almost the same result, so assistant role tokens contribute little. Masking earlier responses raises the harmful share in some cases (meth) and barely changes it in others. Replacing the history with the attack objective raises the harmful share sharply (Figure 4).

**Stated limits.** The probe may pick up differences between the training data and Crescendo text rather than harmfulness. Only Crescendo and Llama-3-8B were studied (Section 5).

**Reading notes for the replication.**
- The 80/20 split is over pooled tokens, so tokens from one response can fall in both splits. Part of the 0.997 and 0.999 accuracy may reflect that. This project splits by example.
- The benign class is retain data, so the probe could be separating content type or style, as the authors acknowledge.
- The strongest effect is on the circuit-breaker model. None of the three target models has circuit breakers, so the original Llama results are the fair comparison. Those are the weaker ones.
- Evidence rests on five manual conversations, no benign multi-turn control and one layer per model.
- Conversations were produced against one model and replayed through both, with the final response held fixed. Activations are therefore compared on identical tokens across k.

## 4. How the three fit together, and the gap

Crescendo shows that gradual escalation defeats aligned models and gives a behavioral account. Arditi et al. show that refusal is mediated by one direction learned from single-turn contrast, and that a single-turn jailbreak can suppress it. Bullwinkel et al. connect the two. A probe trained on single-turn data reads final Crescendo responses as mostly benign, and a single-turn defense does not generalize.

Three questions remain open, and this pilot addresses them.

1. Transfer. The evidence covers one model family at 8B parameters. Whether it holds for Qwen2.5-3B, OLMo-2-1B and a mixture-of-experts model is untested.
2. Value over text. No paper compares an internal probe with an output-only judge on the same conversations, or measures how early each one flags escalation.
3. Evaluation hygiene. Token-level splitting, no benign multi-turn control and a single layer leave the size of the effect uncertain.

## 5. Design consequences for this project

1. **Reference result.** Compare against the original Llama-3-8B-Instruct results, not the circuit-breaker model.
2. **Teacher forcing.** Hold the final response fixed so the same tokens are read at every k, as in the target paper. A model-generated final response changes the token set across k and confounds the comparison. Run generation separately if the unsafe-output check needs it.
3. **Split by conversation.** Group by example id for PCA fitting, probe training and validation. Report per-conversation results, since tokens within a response are not independent.
4. **Controls.** Include the paper's Attack Objective and Single Prompt conditions. Attack Objective also works as a positive control: the probe should flag it on each model before any multi-turn result is trusted.
5. **Matched benign multi-turn conversations.** The paper has none. Without them, the drop from k = 1 to k = 2 cannot be separated from a general effect of adding context.
6. **Layer choice.** Select layers on single-turn validation data only, and report one alternate layer.
7. **Black-box baseline.** Follow the Crescendo judge structure (flag, score, rationale, second review) but score escalation per turn so an earliest-detection turn can be computed. Freeze the prompt before looking at probe results, and handle judge refusals explicitly.
8. **Optional secondary readout.** A difference-in-means direction at post-instruction positions of the final user turn, following Arditi et al., would answer the Week 1 question about the refusal direction directly. It is cheap once activations are cached.
