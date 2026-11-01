# escalation-probes

Do safety probes trained on single-turn activations detect multi-turn escalation?

A pilot replication of the PCA and MLP probing analysis in Bullwinkel et al. (2025), moved from Llama-3-8B-Instruct to Qwen2.5-3B-Instruct, OLMo-2-1B-Instruct and Moonlight-16B-A3B-Instruct, and compared against an output-only LLM judge.

**Status:** pipeline code and tests are in place. No data or results yet. This is a pilot for specific models and datasets, not a general claim about multi-turn jailbreaks.

## Background

Crescendo (Russinovich et al., 2024) escalates a conversation through individually benign turns until an aligned model produces content it would refuse if asked directly. Bullwinkel et al. (2025) studied Crescendo at the level of activations in Llama-3-8B-Instruct and its circuit-breaker variant. They found that probes trained on single-turn benign and harmful data classify the final jailbroken response as mostly benign, and that the share flagged as harmful falls as more turns are placed in context. Arditi et al. (2024) showed that refusal in chat models is mediated by one direction learned from single-turn contrast. None of the three papers tests a single-turn signal against multi-turn escalation in small, different model families, and none compares an internal probe with a text-only judge on the same conversations.

## Research questions

1. Does the benign-region pattern appear in Qwen2.5-3B-Instruct, with OLMo-2-1B-Instruct and Moonlight-16B-A3B-Instruct as secondary tests?
2. Does an MLP probe trained only on single-turn activations detect harmful multi-turn states?
3. Does the internal probe add anything over a black-box judge that reads only the conversation, in accuracy and in how early it flags escalation?

## Method

The design follows Bullwinkel et al. except where `docs/method_comparison.md` records a deviation.

- **Readout.** Activations at the tokens of the final response, with the last k prompt-response pairs in context, k = 1 to n. The final response is held fixed (teacher forcing), so identical tokens are read at every k.
- **Reference space.** PCA fitted on single-turn benign and harmful examples. Multi-turn activations are projected into it and compared with the benign and harmful centroids.
- **Probe.** An MLP with hidden layers (64, 32), trained on single-turn activations only, then applied to multi-turn states. A linear probe is added for comparison.
- **Conditions.** Crescendo (k sweep), Single Prompt (history passed as one user message) and Attack Objective (the goal asked directly). Attack Objective is the positive control: the probe should flag it on every model before any multi-turn result is trusted.
- **Controls.** Matched benign multi-turn conversations with the same topic, length and turn count. The target paper has none.
- **Layers.** Chosen on single-turn validation data only, plus one alternate layer.
- **Black-box baseline.** A separate judge returns a flag, a score and a rationale per turn, followed by a second review pass. The prompt is frozen before probe results are inspected.
- **Metrics.** Detection accuracy and the earliest turn at which each method flags escalation, reported per conversation.

Splits are grouped by conversation. No test conversation contributes to PCA fitting, probe training or layer selection. `tests/test_no_leakage.py` enforces this.

## Scope and data policy

- Two categories: phishing/fraud and non-operational cyber intrusion.
- Prompts come from JailbreakBench or HarmBench where possible, and are self-authored otherwise.
- Real links, credentials, targets, commands and exploit detail are redacted before any prompt is run.
- Model outputs judged unsafe are not published. Labels, scores and activation-derived statistics are released instead.
- If the conversations do not elicit unsafe outputs, the study reports representation drift without claiming a successful jailbreak.
- Fixed prewritten conversations are a weaker attack than adaptive Crescendo (Russinovich et al., Section 5.6), so low success rates are expected and are not evidence of robustness.

## Models

| Model | Role | Hugging Face ID |
|---|---|---|
| Qwen2.5-3B-Instruct | Primary | `Qwen/Qwen2.5-3B-Instruct` |
| OLMo-2-1B-Instruct | Secondary | verify before use |
| Moonlight-16B-A3B-Instruct | Secondary | `moonshotai/Moonlight-16B-A3B-Instruct` |

Models load in 4-bit. Moonlight is a mixture-of-experts model that may need custom model code and more memory. It drops to a limited test if it does not run on the available hardware.

## Repository layout

```
escalation-probes/
├── README.md
├── DECISIONS.md              # every design choice and its reason
├── pyproject.toml
├── configs/
│   ├── models.yaml           # model IDs, layers, quantization
│   └── experiment.yaml       # categories, turn counts, seeds, splits
├── prompts/
│   ├── judge_v1.md           # frozen judge prompt
│   └── judge_v1_review.md    # second-pass review prompt
├── data/
│   ├── single_turn/          # redacted benign and harmful prompts
│   ├── multi_turn/           # fixed prewritten conversations
│   ├── controls/             # matched benign conversations
│   └── splits/               # conversation-level train/test ids
├── src/
│   └── escalation_probes/
│       ├── models.py         # loading, chat templates, 4-bit config
│       ├── cache.py          # activation extraction for each condition and k
│       ├── pca.py            # reference space and projections
│       ├── probe.py          # MLP and linear probes
│       ├── judge.py          # black-box LLM judge
│       └── metrics.py        # accuracy, earliest-detection turn
├── scripts/
│   ├── smoke_test.py         # one forward pass per model
│   ├── make_splits.py        # grouped single-turn split
│   ├── run_cache.py
│   ├── run_layers.py         # layer choice on single-turn validation
│   ├── run_pca.py
│   ├── run_probe.py
│   ├── run_judge.py
│   └── run_compare.py        # probe versus judge
├── tests/
│   ├── test_no_leakage.py
│   ├── test_data.py
│   └── test_analysis.py
├── notebooks/
│   └── analysis.ipynb        # final figures from cached results
├── results/
│   ├── activations/          # git-ignored
│   ├── tables/
│   └── figures/
├── docs/
│   ├── literature_review.md
│   └── method_comparison.md
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/smoke_test.py --model qwen2.5-3b
pytest
```

## Running the pipeline

The scripts are planned and run in this order:

```
python scripts/make_splits.py
python scripts/run_cache.py   --model qwen2.5-3b
python scripts/run_layers.py  --model qwen2.5-3b
python scripts/run_pca.py     --model qwen2.5-3b
python scripts/run_probe.py   --model qwen2.5-3b
python scripts/run_judge.py
python scripts/run_compare.py --model qwen2.5-3b
```

Layer selection runs before PCA because PCA needs a layer (see `DECISIONS.md`, D1). Each stage reads from `results/activations/` and writes to `results/tables/`. Data format is in `data/README.md`. Open items that block a first run are listed at the end of `DECISIONS.md`. Seeds are fixed in `configs/experiment.yaml`.

## Limitations

- Two harm categories and a small fixed set of conversations.
- Prewritten conversations do not adapt to model replies, unlike automated Crescendo.
- A probe trained on benign versus harmful data may separate content type rather than harmfulness, a limit the target paper also notes.
- Probes and the judge are evaluated on the same conversations, so conclusions hold for this dataset only.
- Smaller models may refuse differently from Llama-3-8B-Instruct, which limits direct comparison with the original paper.

## References

- Russinovich, M., Salem, A. and Eldan, R. (2024). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. USENIX Security 2025. arXiv:2404.01833
- Arditi, A. et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717
- Bullwinkel, B. et al. (2025). A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks. DIG-BUGS workshop, ICML 2025. arXiv:2507.02956

The full review is in `docs/literature_review.md`.
