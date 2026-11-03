# Data

All files are JSON Lines, one record per line, and must be redacted before use: no real links, credentials, targets, commands or exploit detail.

`pilot.jsonl` is a synthetic, redacted fixture for validating the pipeline. It
is not a research sample and must not be used for substantive conclusions.

## single_turn_reference/*.jsonl

```json
{"id": "st-b-001", "label": "benign", "category": "phishing_fraud", "prompt": "...", "response": "..."}
```

`label` is `benign` or `harmful`. `response` is the text whose tokens the probe reads. The current configured reference set contains 500 benign and 500 harmful records sampled deterministically from the local public sources, with common links, emails, IP addresses, and code blocks redacted.

The original 16-record fixture remains under `single_turn/` for tests and pipeline smoke checks; it is not included in the experiment config.

## multi_turn/*.jsonl

```json
{"id": "mt-001", "label": "escalation", "category": "cyber_intrusion",
 "objective": "The goal asked directly, for the attack_objective condition",
 "turns": [{"user": "...", "assistant": "..."}, {"user": "...", "assistant": "..."}]}
```

Each turn is an object with `user` and `assistant` fields. Conversations must
alternate implicitly in that order and end with an assistant response.

## controls/*.jsonl

Same schema with `"label": "benign_control"` and `"matched_to": "<escalation id>"`. A control has the same category and turn count as its escalation conversation, and a benign objective on the same topic. `validate_conversations` enforces the match.

## splits/splits.json

Written by `scripts/make_splits.py`: `{"train": [...], "val": [...], "test": [...]}` of single-turn ids. Multi-turn conversations are never in a split because they are test-only.
