"""Validate conversation files before any model run."""
import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

required = {"id", "label", "category", "objective", "turns"}
categories = {"phishing_fraud", "cyber_intrusion"}
labels = {"escalation", "benign_control"}
leaks = {
    "url": re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|io|ly)/", re.I),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "ipv4": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "code_fence": re.compile(r"```"),
}


def load(paths):
    """Read every JSONL record from files, directories or glob patterns."""
    records = []
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            matches = [Path(match) for match in glob(raw_path)]
            files.extend(matches or [path])
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"no JSONL file found for {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
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
    if not isinstance(rec["turns"], list) or not rec["turns"]:
        errors.append(f"{rid}: turns must be a non-empty list")
        return errors
    for turn in rec["turns"]:
        if not isinstance(turn, dict) or {"user", "assistant"} - turn.keys():
            errors.append(f"{rid}: turns need user and assistant text")
            continue
        for content in (turn["user"], turn["assistant"]):
            for name, pattern in leaks.items():
                if pattern.search(content):
                    errors.append(f"{rid}: {name} pattern found, redact it")
    return errors


def words(rec):
    """Return per-turn word counts."""
    return [len(f"{turn['user']} {turn['assistant']}".split()) for turn in rec["turns"]]


def check_groups(records):
    """Check matched control references and per-turn length matching."""
    errors = []
    by_id = {rec["id"]: rec for rec in records}
    pairs = defaultdict(list)
    for rec in records:
        if rec["label"] == "benign_control":
            matched_to = rec.get("matched_to")
            target = by_id.get(matched_to)
            if target is None or target["label"] != "escalation":
                errors.append(f"{rec['id']}: matched_to must reference an escalation")
                continue
            pairs[matched_to].append(rec)
    for target_id, controls in pairs.items():
        if len(controls) != 1:
            errors.append(f"{target_id}: needs exactly one benign_control")
            continue
        harmful = words(by_id[target_id])
        control = words(controls[0])
        if len(harmful) != len(control):
            errors.append(f"{target_id}: turn counts differ")
            continue
        for index, (a, b) in enumerate(zip(harmful, control), start=1):
            if abs(a - b) > 0.2 * max(a, b, 1):
                errors.append(f"{target_id}: turn {index} length differs by more than 20%")
    return errors


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    records = load(args.files)
    errors = [err for rec in records for err in check_record(rec)]
    errors += check_groups(records)
    if not records:
        errors.append("no records found")
    for err in errors:
        log.error(err)
    log.info("%d records, %d errors", len(records), len(errors))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
