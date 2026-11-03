# escalation-probes

Do safety probes trained on single-turn activations detect multi-turn escalation?

A pilot replication of the PCA and MLP probing analysis in Bullwinkel et al. (2025), moved from Llama-3-8B-Instruct to Qwen2.5-3B-Instruct, OLMo-2-1B-Instruct and Moonlight-16B-A3B-Instruct, and compared against an output-only LLM judge.

**Status:** the Qwen primary-model pilot has completed caching, PCA, layer selection, and probe evaluation. The repository has a synthetic redacted fixture dataset, schema validation, grouped splits, logging, progress bars, and 28 passing tests. This is a pipeline validation run, not evidence from a substantive research dataset.

## Current project status

```mermaid
flowchart LR
    A[Plan and literature review] --> B[Repository setup]
    B --> C[Redacted fixture data]
    C --> D[Validation and grouped splits]
    D --> E[Qwen smoke test]
    E -->     F[Activation cache<br/>104 / 104 files] --> G[Layer selection]
    G --> H[PCA reference space]
    H --> I[MLP and linear probes]
    I --> J[Optional output-only judge]
    J --> K[Comparison and report]

    classDef done fill:#d1fae5,stroke:#059669,color:#064e3b;
    classDef active fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef next fill:#e5e7eb,stroke:#6b7280,color:#111827;
    class A,B,C,D,E,F,G,H,I done;
    class J active;
    class K next;
```

### Completed

- Read and incorporated the individual plan and core literature notes.
- Added repository steering instructions in `.github/copilot-instructions.md`.
- Installed the project in the local `.venv`.
- Added a deterministic public-dataset reference set: 500 benign and 500 harmful single-turn records.
- Kept the original synthetic fixture separately for tests and smoke checks.
- Added schema, redaction, control-matching, and leakage checks.
- Added deterministic grouped train/validation/test splits for the 1,000-record reference set.
- Added progress bars and logging to all pipeline scripts.
- Verified the test suite: **28 passed**.
- Completed the Qwen model smoke test, activation cache, layer selection, PCA, and probe evaluation.

### Current checkpoint

- Primary model: `Qwen/Qwen2.5-3B-Instruct`
- Reference set: 500 benign + 500 harmful public-dataset records
- OLMo secondary model: `allenai/OLMo-2-0425-1B-Instruct` (verified in configuration)
- Loading mode: 4-bit quantization
- Candidate layers: 6
- Previous fixture cache: 104 files
- Scaled reference cache: not yet generated
- Generated tables: PCA variance, PCA centroids, single-turn probe results, multi-turn probe results, and selected layers
- Cached arrays contain final response-token activations only; the redacted conversations remain readable in `data/`.

### Preliminary Qwen fixture results

These are pipeline checks on the synthetic fixture data, not research conclusions:

- Selected layer: `11`; alternate layer: `18`.
- Validation token accuracy was `1.00` for every candidate layer.
- Selected-layer MLP test token accuracy: `0.907`; linear probe: `0.930`.
- On the selected-layer MLP, all four escalation fixtures first crossed the 0.5 harmful-token threshold at turn 3.
- Final full-history harmful-token shares were `0.95`, `1.00`, `0.95`, and `0.895` for the four escalation fixtures.
- Three of four benign controls had zero harmful-token share at the final turn; one had `0.125`.
- The first two PCA components explained `14.6%` at the selected layer and `15.9%` at the alternate layer.

The earlier fixture was intentionally small and synthetic; those numbers only confirmed that the analysis stages produce outputs. The scaled reference run must be regenerated from `run_cache.py` before interpreting new PCA or probe results.

The local source files and derived 1,000-record reference set are excluded from Git because they may contain sensitive or unsafe text. Their provenance and deterministic conversion are recorded in `scripts/prepare_reference_data.py`.

### Qwen visualizations

The full paper-style, step-by-step analysis is in [`notebooks/qwen_paper_replication_analysis.ipynb`](notebooks/qwen_paper_replication_analysis.ipynb). It reconstructs the PCA reference-space analysis, centroid geometry, harmful-token share versus `k`, per-turn detection, condition comparisons, controls, and layer sensitivity.

The notebook writes these visualizations from the completed Qwen fixture tables:

- [Harmful-token share by turn](results/figures/qwen_harmful_share_by_turn.png)
- [PCA cumulative variance](results/figures/qwen_pca_variance.png)
- [Final-turn harmful-token share](results/figures/qwen_final_turn_harmful_share.png)
- [Paper-style PCA variance](results/figures/qwen_paper_pca_variance.png)
- [Paper-style centroid geometry](results/figures/qwen_paper_single_turn_centroid_geometry.png)
- [Paper-aligned Figure 1: full conversation vs attack objective](results/figures/qwen_paper_figure1_rq1_fc_vs_ao.png)
- [Paper-aligned Figure 1: `k=1` vs `k>1`](results/figures/qwen_paper_figure1_rq2_k1_vs_kgt1.png)
- [Paper-aligned Figure 2: MLP summary](results/figures/qwen_paper_figure2_mlp_summary.png)
- [Paper-style harmful share by `k`](results/figures/qwen_paper_harmful_share_by_k.png)
- [Paper-style per-turn detection](results/figures/qwen_paper_per_turn_detection.png)
- [Paper-style condition comparison](results/figures/qwen_paper_condition_comparison.png)
- [Paper-style layer sensitivity](results/figures/qwen_paper_layer_sensitivity.png)

### Next commands

After `run_cache.py` reaches 104 files, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_layers.py --model qwen2.5-3b
.\.venv\Scripts\python.exe scripts\run_pca.py --model qwen2.5-3b
.\.venv\Scripts\python.exe scripts\run_probe.py --model qwen2.5-3b
```

The external judge is optional and requires a separately configured API-backed model. OLMo and Moonlight are secondary models and are not required to complete the first Qwen pilot.

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
| OLMo-2-1B-Instruct | Secondary | `allenai/OLMo-2-0425-1B-Instruct`  |
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
