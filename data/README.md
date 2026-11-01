# Data

All files are JSON Lines, one record per line, and must be redacted before use: no real links, credentials, targets, commands or exploit detail.

## single_turn/*.jsonl

```json
{"id": "st-b-001", "label": "benign", "category": "phishing_fraud", "prompt": "...", "response": "..."}
```

`label` is `benign` or `harmful`. `response` is the text whose tokens the probe reads. See open item 1 in `DECISIONS.md` for where harmful responses come from.

## multi_turn/*.jsonl

```json
{"id": "mt-001", "label": "escalation", "category": "cyber_intrusion",
 "objective": "The goal asked directly, for the attack_objective condition",
 "turns": [{"user": "...", "assistant": "..."}, {"user": "...", "assistant": "..."}]}
```

## controls/*.jsonl

Same schema with `"label": "benign_control"` and `"matched_to": "<escalation id>"`. A control has the same category and turn count as its escalation conversation, and a benign objective on the same topic. `validate_conversations` enforces the match.

## splits/splits.json

Written by `scripts/make_splits.py`: `{"train": [...], "val": [...], "test": [...]}` of single-turn ids. Multi-turn conversations are never in a split because they are test-only.
