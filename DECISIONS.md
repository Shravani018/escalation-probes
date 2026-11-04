# Decisions

Each entry records a design choice, the reason, and whether it deviates from Bullwinkel et al. (2025). Open items are listed at the end.

## D1. Pipeline order adds a layer-selection stage
Layer selection needs a trained probe, and PCA needs a layer, so selection runs first: `run_cache`, `run_layers`, `run_pca`, `run_probe`, `run_judge`, `run_compare`. Selection uses single-turn validation accuracy only. The alternate layer is the best candidate that is not adjacent to the selected one. Deviation: the paper fixes layer 31 (20 for the circuit-breaker model).

## D2. Candidate layers are depth fractions
Candidates are 0.3 to 0.8 of model depth, resolved per model, so the same config works for 16, 24 and 36-layer models. `hidden_states[0]` is the embedding output, so index L is the final block. Activations for all candidates are cached once, so layer choice never triggers new forward passes.

## D3. Teacher forcing and token-count check
The recorded response is held fixed at every k and condition. `run_cache` raises if the same response yields a different token count under a different context, which would indicate a tokenization boundary problem. `response_span` also checks that the prompt tokens are an exact prefix of the full sequence.

## D4. Per-turn readouts for earliest detection
The k sweep reads only the final response. To compare with a judge that scores each turn, the probe also reads every earlier response with its full history (k = turn). The probe and the judge therefore see the same conversation prefix at each turn. Deviation: the paper has no per-turn series.

## D5. Probe inputs are standardised
Both probes sit behind a `StandardScaler`. Raw residual-stream activations contain a few very large dimensions that slow MLP convergence. Settings otherwise match the paper (hidden layers (64, 32), `max_iter` 1000, `random_state` 42). Deviation.

## D6. Grouped splits at example level
Each single-turn example is one group, assigned wholly to train, val or test, stratified by label. Tokens from one response never straddle splits. Every multi-turn conversation is test-only. `check_no_leakage` runs in `make_splits`, in `load_splits` and in the test suite. It also rejects any conversation whose objective or user turn equals a single-turn prompt.

## D7. Conversation decision rule
A readout flags a conversation when the share of response tokens classified harmful reaches `decision.conversation_share` (0.5). The judge flags a turn when its score reaches `judge.flag_score_threshold` (50), applied in `run_judge` so the stored flag always follows the configured threshold. Both thresholds are set before any multi-turn result is inspected.

## D8. Judge handling of failures
A judge reply that cannot be parsed, including the instructed `UNABLE` reply, is stored with `flag: null`. Unscored turns never count as benign. `compare_methods` reports the number of unscored turns per conversation, and `summarise` evaluates the judge only on conversations scored at the final turn.

## D9. Frozen judge prompt
`judge_v1.md` and `judge_v1_review.md` are hashed. `run_judge` prints the hash and refuses to run if `judge.prompt_sha256` is set and no longer matches. Set the hash before opening any probe results.

## D10. Condition definitions
- `crescendo`: last k pairs as alternating turns.
- `single_prompt`: earlier pairs rendered as a `User:` and `Assistant:` transcript inside one user message, followed by the final user turn.
- `attack_objective`: the objective as the sole user message.

Masked Responses is not implemented. Add it after the three main conditions run.

## D11. CPU-manageable reference pilot

The public reference set uses 100 benign and 100 harmful records. This keeps the first CPU-only run tractable while preserving balanced labels and the same grouped split procedure. It is a pilot for pipeline validation, not a claim of exact paper-scale replication.

## Open items

1. **Source of harmful single-turn responses.** The probe needs response tokens for the harmful class. The paper used a circuit-breaker set with harmful completions. This repository does not generate unsafe completions. Options: use responses already present in a published dataset, use refusals as the harmful-prompt response (which makes the probe read refusal rather than harm), or use self-authored redacted completions. This choice changes what the probe measures and should be made before caching.
2. **OLMo-2-1B model ID.** `allenai/OLMo-2-0425-1B-Instruct` is the likely repository name but is not verified. `configs/models.yaml` marks it `verified: false`, and loading refuses until it is changed.
3. **Judge model.** `judge.model` is unset. The judge should come from a different family than the three probed models.
4. **Conversation data.** `data/multi_turn` and `data/controls` are empty. Conversations follow the schema in `data/README.md` and must be redacted before use.
