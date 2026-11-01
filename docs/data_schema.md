# Data schema and redaction rules

Conversation files are JSONL, one record per line, under `data/`.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | string | unique record id |
| `group_id` | string | conversation family; all turn-prefixes of one conversation share it |
| `pair_id` | string | links a conversation to its matched benign control |
| `split` | `train` or `test` | assigned by `group_id`, never by turn or token |
| `category` | `phishing_fraud` or `cyber_intrusion` | study category |
| `label` | `harmful` or `benign_control` | conversation type |
| `source` | `jailbreakbench`, `harmbench` or `self_authored` | origin of the seed prompt |
| `turns` | list of `{role, content}` | alternating user and assistant, starting with user, ending with assistant |
| `redaction_checked` | bool | set true only after manual review |

## Redaction

Replace the following with placeholders before any prompt is run:

| Item | Placeholder |
|---|---|
| links and domains | `[LINK]` |
| credentials, tokens, account numbers | `[CREDENTIAL]` |
| named people, organizations, systems | `[TARGET]` |
| commands, code, exploit detail | `[REDACTED]` |

The target paper's phishing criteria require a specific link. Here the link stays a placeholder, so success is judged on structure and intent, not on a working artifact.

## Sourcing

1. Seed prompts come from JailbreakBench or HarmBench behaviors in the two categories.
2. Multi-turn escalation conversations are not generated in this repository's tooling. They come from published examples (the target paper's appendix), adapted and redacted, or from researcher-authored text reviewed with the supervisor.
3. Anything authored goes through the redaction table above before it is saved.

## Matched benign controls

Each harmful conversation has one control with the same `pair_id` that matches on:

- turn count
- per-turn word count, within 20%
- general topic area, kept at the level of everyday knowledge
- pattern of increasing detail, directed at a harmless end

## Checks

`validate_data.py` enforces the schema, the redaction patterns, role alternation, split disjointness by `group_id` and control matching. Run it before every caching job.
