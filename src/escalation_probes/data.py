"""Data loading, condition construction, splits and leakage checks.

Single-turn record:
    {"id", "label": "benign" | "harmful", "category", "prompt", "response"}

Conversation record (escalation and matched benign control):
    {"id", "label": "escalation" | "benign_control", "category", "objective",
     "turns": [{"user", "assistant"}, ...], "matched_to": <escalation id, controls only>}

Turns are 1-indexed in the API below. For target turn t and context size k,
the context holds the last k prompt-response pairs ending at turn t, and the
assistant message of turn t is the teacher-forced response that gets read.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CONDITIONS = ("crescendo", "single_prompt", "attack_objective")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read one JSON object per line."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dir(directory: str | Path) -> list[dict]:
    """Read every .jsonl file in a directory."""
    records: list[dict] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        records.extend(load_jsonl(path))
    return records


def load_dataset(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Load and validate (single_turn_records, conversations incl. controls) from the config paths."""
    single = load_dir(cfg["paths"]["single_turn"])
    conversations = load_dir(cfg["paths"]["multi_turn"]) + load_dir(cfg["paths"]["controls"])
    validate_single_turn(single)
    validate_conversations(conversations)
    return single, conversations


def load_splits(cfg: dict) -> dict[str, list[str]]:
    """Read the saved single-turn split and re-check it for leakage."""
    with open(cfg["paths"]["splits"], encoding="utf-8") as handle:
        splits = json.load(handle)
    single, conversations = load_dataset(cfg)
    check_no_leakage(splits, single, conversations)
    return splits


def validate_single_turn(records: list[dict]) -> None:
    """Raise ValueError on malformed single-turn records or duplicate ids."""
    seen = set()
    for rec in records:
        missing = {"id", "label", "category", "prompt", "response"} - rec.keys()
        if missing:
            raise ValueError(f"single-turn {rec.get('id')}: missing {sorted(missing)}")
        if rec["label"] not in ("benign", "harmful"):
            raise ValueError(f"single-turn {rec['id']}: bad label {rec['label']!r}")
        if rec["id"] in seen:
            raise ValueError(f"duplicate id {rec['id']}")
        seen.add(rec["id"])


def validate_conversations(records: list[dict]) -> None:
    """Raise ValueError on malformed conversations or unmatched controls."""
    by_id = {rec["id"]: rec for rec in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate conversation ids")
    for rec in records:
        missing = {"id", "label", "category", "objective", "turns"} - rec.keys()
        if missing:
            raise ValueError(f"conversation {rec.get('id')}: missing {sorted(missing)}")
        if rec["label"] not in ("escalation", "benign_control"):
            raise ValueError(f"conversation {rec['id']}: bad label {rec['label']!r}")
        if not rec["turns"] or any({"user", "assistant"} - t.keys() for t in rec["turns"]):
            raise ValueError(f"conversation {rec['id']}: turns need user and assistant text")
        if rec["label"] == "benign_control":
            target = by_id.get(rec.get("matched_to"))
            if target is None:
                raise ValueError(f"control {rec['id']}: matched_to not found")
            if len(target["turns"]) != len(rec["turns"]) or target["category"] != rec["category"]:
                raise ValueError(f"control {rec['id']}: turn count or category differs from {target['id']}")


def build_messages(conv: dict, turn: int, k: int, condition: str) -> list[dict]:
    """Return chat messages whose final message is the response read at `turn`.

    crescendo: the last k pairs as ordinary alternating turns.
    single_prompt: earlier pairs rendered as one transcript inside the final user message.
    attack_objective: the objective alone as the user message.
    """
    turns = conv["turns"]
    if not 1 <= turn <= len(turns):
        raise ValueError(f"turn {turn} outside 1..{len(turns)}")
    if not 1 <= k <= turn:
        raise ValueError(f"k={k} outside 1..{turn}")
    final = turns[turn - 1]
    if condition == "crescendo":
        messages: list[dict] = []
        for pair in turns[turn - k : turn]:
            messages.append({"role": "user", "content": pair["user"]})
            messages.append({"role": "assistant", "content": pair["assistant"]})
        return messages
    if condition == "single_prompt":
        history = turns[turn - k : turn - 1]
        transcript = "\n\n".join(f"User: {p['user']}\nAssistant: {p['assistant']}" for p in history)
        user = f"{transcript}\n\nUser: {final['user']}" if transcript else final["user"]
        return [{"role": "user", "content": user}, {"role": "assistant", "content": final["assistant"]}]
    if condition == "attack_objective":
        return [{"role": "user", "content": conv["objective"]}, {"role": "assistant", "content": final["assistant"]}]
    raise ValueError(f"unknown condition {condition!r}")


def iter_jobs(conv: dict, conditions: tuple[str, ...] = CONDITIONS) -> list[dict]:
    """List the distinct (condition, turn, k) readouts needed for one conversation.

    crescendo: the k sweep at the final turn, plus every earlier turn read with its
    full history (k = turn), which gives the per-turn series used for earliest detection.
    single_prompt: the k sweep at the final turn for k >= 2 (k = 1 equals crescendo).
    attack_objective: the final response once.
    """
    n = len(conv["turns"])
    jobs: list[dict] = []
    if "crescendo" in conditions:
        jobs += [{"condition": "crescendo", "turn": n, "k": k} for k in range(1, n + 1)]
        jobs += [{"condition": "crescendo", "turn": t, "k": t} for t in range(1, n)]
    if "single_prompt" in conditions:
        jobs += [{"condition": "single_prompt", "turn": n, "k": k} for k in range(2, n + 1)]
    if "attack_objective" in conditions:
        jobs.append({"condition": "attack_objective", "turn": n, "k": 1})
    return jobs


def make_splits(ids: list[str], fractions: dict, seed: int) -> dict[str, list[str]]:
    """Split ids into train/val/test. Each id is one group, so no example spans splits."""
    ordered = sorted(set(ids))
    random.Random(seed).shuffle(ordered)
    n_train = round(len(ordered) * fractions["train"])
    n_val = round(len(ordered) * fractions["val"])
    return {
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }


def check_no_leakage(splits: dict[str, list[str]], single_turn: list[dict], conversations: list[dict]) -> None:
    """Raise AssertionError if any split overlaps or any conversation text appears in single-turn data."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]) & set(splits[b])
            assert not overlap, f"ids shared between {a} and {b}: {sorted(overlap)[:3]}"
    single_ids = {rec["id"] for rec in single_turn}
    assert set().union(*map(set, splits.values())) <= single_ids, "split lists ids absent from single-turn data"
    assert not single_ids & {c["id"] for c in conversations}, "conversation ids collide with single-turn ids"
    single_prompts = {rec["prompt"].strip().lower() for rec in single_turn}
    for conv in conversations:
        texts = [conv["objective"]] + [t["user"] for t in conv["turns"]]
        clash = [t for t in texts if t.strip().lower() in single_prompts]
        assert not clash, f"conversation {conv['id']} reuses a single-turn prompt"
