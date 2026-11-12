# escalation-probes

Do safety probes trained on single-turn activations detect multi-turn escalation?

A method-faithful pilot replication of the PCA and MLP probing analysis in Bullwinkel et al. (2025), evaluated across Qwen2.5-3B-Instruct, OLMo-2-1B-Instruct, and SmolLM2-1.7B-Instruct, with an optional output-only LLM judge.

**Status:** the single-turn pipeline (activation cache, layer selection, PCA, probes) is complete for Qwen, OLMo and SmolLM2. Round one of the multi-turn test used hand-written refusal replies, so it cannot test the paper's central claim and is kept as pilot evidence only. Round two (`data/prompts_v2`, 50 escalation and 50 matched control conversations with real Qwen replies, plus a free hosted judge) replaces it. This is method validation under a local compute budget, not an exact replication.

## Current project status

```mermaid
flowchart LR
    A[Plan and literature review] --> B[Repository setup]
    B --> C[Redacted fixture data]
    C --> D[Validation and grouped splits]
    D --> E[Qwen smoke test]
    E -->     F[Activation cache<br/>200-record pilot] --> G[Layer selection]
    G --> H[PCA reference space]
    H -->     I[MLP and linear probes]
    I --> J[Optional output-only judge]
    J --> K[Comparison and report]

    classDef done fill:#d1fae5,stroke:#059669,color:#064e3b;
    classDef active fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef next fill:#e5e7eb,stroke:#6b7280,color:#111827;
    class A,B,C,D,E,F,G,H,I,J,K done;
```

### Completed

- Read and incorporated the individual plan and core literature notes.
- Added repository steering instructions in `.github/copilot-instructions.md`.
- Installed the project in the local `.venv`.
- Added a deterministic public-dataset reference set: 100 benign and 100 harmful single-turn records.
- Kept the original synthetic fixture separately for tests and smoke checks.
- Added schema, redaction, control-matching, and leakage checks.
- Added deterministic grouped train/validation/test splits for the 200-record reference set.
- Added progress bars and logging to all pipeline scripts.
- Verified the test suite: **28 passed**.
- Completed the Qwen, OLMo, and SmolLM2 activation caches, layer selection, PCA, and probe evaluation.
- Completed the 32-row local Ollama judge run and probe-versus-judge comparisons for all three models.
- Added and executed the all-models paper-style notebook with labeled PCA, trajectory, condition, layer, and judge-comparison figures.

### Current checkpoint

- Primary model: `Qwen/Qwen2.5-3B-Instruct`
- Reference set: 200 records: 100 benign + 100 harmful public-dataset records
- Secondary models: `allenai/OLMo-2-0425-1B-Instruct` and `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Loading mode: 4-bit quantization
- Candidate layers: 6
- Previous fixture cache remains alongside the current cache
- Current reference run: 200 records, split 140/30/30, completed for all three models
- Generated tables: PCA variance, PCA centroids, single-turn probe results, multi-turn probe results, and selected layers
- Cached arrays contain final response-token activations only; the redacted conversations remain readable in `data/`.

### Round-one pilot results (hand-written refusal replies)

The table reports selected-layer MLP performance on held-out single-turn tokens and conversation-level final-flag accuracy on the eight multi-turn conversations (four escalations and four matched controls). Judge accuracy is the same text-only judge for every model; `n=8` is the number of conversations, not the number of single-turn records.

| Model | Selected / alternate layer | MLP test token accuracy | MLP test example accuracy | Probe conversation accuracy | Judge accuracy | Judge scored |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | 25 / 18 | 94.2% | 96.7% | 75.0% | 37.5% | 8/8 |
| OLMo-2-1B-Instruct | 8 / 11 | 91.8% | 96.7% | 62.5% | 37.5% | 8/8 |
| SmolLM2-1.7B-Instruct | 17 / 12 | 93.6% | 100.0% | 50.0% | 37.5% | 8/8 |

Selected-layer PCA cumulative variance after two components was 7.3% for Qwen, 7.5% for OLMo, and 8.3% for SmolLM2. These values describe the representation geometry; they are not accuracy scores.

### Individual-plan coverage

| Planned component | Status | Evidence |
|---|---|---|
| Three-model comparison | Complete with documented substitution | Qwen, OLMo, and SmolLM2 cache, layer, PCA, and probe tables; SmolLM2 replaces Moonlight-16B-A3B, which exceeds the local compute budget |
| Single-turn grouped train/validation/test evaluation | Complete | 200 records, split 140/30/30, with leakage checks |
| Multi-turn escalation and matched controls | Complete | 4 escalation conversations plus 4 controls, four turns each |
| PCA reference space and centroid analysis | Complete | `pca_variance_*.csv`, `pca_centroids_*.csv`, notebook Figures 2 and 7 |
| MLP and linear probe analysis | Complete | `probe_single_turn_*.csv`, `probe_multi_turn_*.csv` |
| Text-only judge comparison | Complete | 32 judge turn rows and `comparison_*.csv` / `summary_*.csv` |
| Reproducible prompts and configuration | Complete | Frozen prompt hash in `configs/experiment.yaml` |
| Paper-style visual reporting | Complete | `notebooks/all_models_paper_analysis.ipynb` |
| Rerun weak or ambiguous cases | Not performed | Results are locked as an exploratory pilot; no additional seeds or adaptive conversations were run |
| Final presentation and blog | Not in repository | Required by the individual plan but remains a writing deliverable |

The round-one analysis is complete. Round two (below) is the remaining model run, followed by the write-up, presentation and blog. The pilot is limited by four escalation conversations, four controls, prewritten rather than adaptive turns, and a local supplementary judge. It does not support broad claims of generalization or robustness.

### Judge prompts and inputs

The optional text-only judge reads every conversation from `data/multi_turn/*.jsonl` and `data/controls/*.jsonl`, including the four escalation conversations and their four matched controls in the current pilot. For each turn, it sends the full conversation history through the current turn to the judge, then sends the first verdict for review.

The judge prompts are:

- `prompts/judge_v1.md`: escalation-monitor instructions and JSON verdict format.
- `prompts/judge_v1_review.md`: second-pass verdict review instructions.

Judge outputs are written to `results/tables/judge_turns.jsonl`. The judge does not read PCA files, probe activations, or single-turn reference records.

### Qwen pilot conclusion

The Qwen2.5-3B probe learned a strong single-turn benign/harmful separation: selected-layer MLP test token accuracy was 94.2%. On the held-out multi-turn pilot, escalation conversations averaged 78.2% harmful-token share versus 49.6% for matched benign controls, while the direct Attack Objective control averaged 84.5%. The result supports the feasibility of the probe, but the control false-positive rate is too high to claim reliable escalation detection.

**One takeaway:** the probe detects a broad harmfulness-related representation, but this pilot does not yet show that it isolates multi-turn escalation.

The local source files and derived 200-record reference set are excluded from Git because they may contain sensitive or unsafe text. Their provenance and deterministic conversion are recorded in `scripts/prepare_reference_data.py`.

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

### Completed pipeline commands

After `run_cache.py` completes the current reference run, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_layers.py --model qwen2.5-3b
.\.venv\Scripts\python.exe scripts\run_pca.py --model qwen2.5-3b
.\.venv\Scripts\python.exe scripts\run_probe.py --model qwen2.5-3b
```

The text-only judge used the locally configured Ollama model `llama3.2:3b`.

The all-models paper-style analysis notebook is [`notebooks/all_models_paper_analysis.ipynb`](notebooks/all_models_paper_analysis.ipynb). It produces indexed figures for single-turn accuracy, PCA variance and scatter geometry, context length, per-turn and category trajectories, condition/layer sensitivity, and probe-versus-judge comparison.

## Round two: real replies and a free judge

Round one showed that hand-written refusals give the probe nothing to detect. Round two generates the replies.

```
python scripts/build_prompts.py                                  # writes data/prompts_v2 (50 + 50 user-turn sets)
python scripts/generate_replies.py --model qwen2.5-3b            # free generation, greedy, resumable
python scripts/label_outputs.py                                  # manual refused / partial / complied check
python scripts/run_cache.py --model qwen2.5-3b                   # single-turn files are skipped
python scripts/run_pca.py --model qwen2.5-3b
python scripts/run_probe.py --model qwen2.5-3b
mv results/tables/judge_turns.jsonl results/tables/judge_turns_llama3b.jsonl
python scripts/run_judge.py                                      # needs GROQ_API_KEY, see configs/experiment.yaml
python scripts/run_compare.py --model qwen2.5-3b
```

`run_layers.py` is not rerun: layers depend on single-turn validation data only and stay at the values in `selected_layers.json`.
Generated conversations and labels are git-ignored because they may contain unsafe text. If the labels show no partial or complied replies, report the study as a representation-monitoring pilot without a jailbreak claim.

## AI assistance

Claude was used for timeline generation, for drafting the round-two prompt templates and scripts, and for code review. GitHub Copilot steering instructions were used during repository setup. All results come from the code in this repository.

## Background

Crescendo (Russinovich et al., 2024) escalates a conversation through individually benign turns until an aligned model produces content it would refuse if asked directly. Bullwinkel et al. (2025) studied Crescendo at the level of activations in Llama-3-8B-Instruct and its circuit-breaker variant. They found that probes trained on single-turn benign and harmful data classify the final jailbroken response as mostly benign, and that the share flagged as harmful falls as more turns are placed in context. Arditi et al. (2024) showed that refusal in chat models is mediated by one direction learned from single-turn contrast. None of the three papers tests a single-turn signal against multi-turn escalation in small, different model families, and none compares an internal probe with a text-only judge on the same conversations.

## Research questions

1. Does the benign-region pattern appear across Qwen2.5-3B-Instruct, OLMo-2-1B-Instruct, and SmolLM2-1.7B-Instruct?
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
| Qwen2.5-3B-Instruct | Primary · 2024 | `Qwen/Qwen2.5-3B-Instruct` |
| OLMo-2-1B-Instruct | Secondary · 2025 | `allenai/OLMo-2-0425-1B-Instruct` |
| SmolLM2-1.7B-Instruct | Secondary · 2024 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

All models load in 4-bit. The secondary models are intentionally small enough for the local CPU workflow, although caching remains the slowest stage.

## Repository layout

```
escalation-probes/
├── README.md
├── DECISIONS.md              # local methodological notes (Git-ignored)
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
│   ├── all_models_paper_analysis.ipynb
│   └── qwen_paper_replication_analysis.ipynb
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
.\.venv\Scripts\python.exe scripts\run_cache.py --model qwen2.5-3b --log-file qwen_run.txt
python scripts/run_layers.py  --model qwen2.5-3b
python scripts/run_pca.py     --model qwen2.5-3b
python scripts/run_probe.py   --model qwen2.5-3b
.\.venv\Scripts\python.exe scripts\run_cache.py --model olmo-2-1b --log-file olmo_run.txt
python scripts/run_layers.py  --model olmo-2-1b
python scripts/run_pca.py     --model olmo-2-1b
python scripts/run_probe.py   --model olmo-2-1b
.\.venv\Scripts\python.exe scripts\run_cache.py --model smollm2-1.7b --log-file smollm2_run.txt
.\.venv\Scripts\python.exe scripts\run_layers.py --model smollm2-1.7b
.\.venv\Scripts\python.exe scripts\run_pca.py --model smollm2-1.7b
.\.venv\Scripts\python.exe scripts\run_probe.py --model smollm2-1.7b
python scripts/run_judge.py
python scripts/run_compare.py --model qwen2.5-3b
```

Layer selection runs before PCA because PCA needs a layer (see `DECISIONS.md`, D1). Each stage reads from `results/activations/` and writes to `results/tables/`. Data format is in `data/README.md`. Open items that block a first run are listed at the end of `DECISIONS.md`. Seeds are fixed in `configs/experiment.yaml`.

## Limitations

- Two harm categories and a small fixed set of conversations.
- Prewritten conversations do not adapt to model replies, unlike automated Crescendo.
- A probe trained on benign versus harmful data may separate content type rather than harmfulness, a limit the target paper also notes.
- Probes and the judge are evaluated on the same conversations, so conclusions hold for this dataset only.
- Smaller models and different chat templates may refuse differently from Llama-3-8B-Instruct, which limits direct comparison with the original paper.

## References

- Russinovich, M., Salem, A. and Eldan, R. (2024). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. USENIX Security 2025. arXiv:2404.01833
- Arditi, A. et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717
- Bullwinkel, B. et al. (2025). A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks. DIG-BUGS workshop, ICML 2025. arXiv:2507.02956

The full review is in `docs/literature_review.md`.
