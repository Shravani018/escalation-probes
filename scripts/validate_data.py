"""Validate conversation files before any model run."""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

required = {"id", "group_id", "pair_id", "split", "category", "label", "source", "turns"}
categories = {"phishing_fraud", "cyber_intrusion"}
labels = {"harmful", "benign_control"}
sources = {"jailbreakbench", "harmbench", "self_authored"}
leaks = {
    "url": re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|io|ly)/", re.I),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "ipv4": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "code_fence": re.compile(r"```"),
}


def load(paths):
    """Read every JSONL record from the given files."""
    records = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def check_record(rec):
    """Return schema, redaction and role errors for one record."""
    errors = []
    missing = required - rec.keys()
    if missing:
        return [f"{rec.get('id', '?')}: missing {sorted(missing)}"]
    rid = rec["id"]
    if rec["category"] not in categories:
        errors.append(f"{rid}: bad category")
    if rec["label"] not in labels:
        errors.append(f"{rid}: bad label")
    if rec["source"] not in sources:
        errors.append(f"{rid}: bad source")
    if rec["split"] not in {"train", "test"}:
        errors.append(f"{rid}: bad split")
    roles = [turn["role"] for turn in rec["turns"]]
    expected = ["user", "assistant"] * (len(roles) // 2)
    if not roles or roles != expected:
        errors.append(f"{rid}: roles must alternate user/assistant and end with assistant")
    for turn in rec["turns"]:
        for name, pattern in leaks.items():
            if pattern.search(turn["content"]):
                errors.append(f"{rid}: {name} pattern found, redact it")
    return errors


def words(rec):
    """Return per-turn word counts."""
    return [len(turn["content"].split()) for turn in rec["turns"]]


def check_groups(records):
    """Check split disjointness by group and control matching by pair."""
    errors = []
    splits = defaultdict(set)
    pairs = defaultdict(dict)
    for rec in records:
        splits[rec["group_id"]].add(rec["split"])
        pairs[rec["pair_id"]][rec["label"]] = rec
    for group, found in splits.items():
        if len(found) > 1:
            errors.append(f"group {group}: appears in both splits")
    for pair, members in pairs.items():
        if set(members) != labels:
            errors.append(f"pair {pair}: needs one harmful and one benign_control")
            continue
        harmful, control = words(members["harmful"]), words(members["benign_control"])
        if len(harmful) != len(control):
            errors.append(f"pair {pair}: turn counts differ")
            continue
        for index, (a, b) in enumerate(zip(harmful, control), start=1):
            if abs(a - b) > 0.2 * max(a, b, 1):
                errors.append(f"pair {pair}: turn {index} length differs by more than 20%")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    records = load(args.files)
    errors = [err for rec in records for err in check_record(rec)]
    errors += check_groups(records)
    for err in errors:
        print(err)
    print(f"{len(records)} records, {len(errors)} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
