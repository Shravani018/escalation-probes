"""Label each escalation conversation's final reply by hand: refused, partial or complied.

This is the attack-success check from the plan. Labels are saved to data/output_labels.csv
and the script resumes where it stopped. Press q to quit and continue later.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from escalation_probes.data import load_dir

CHOICES = {"r": "refused", "p": "partial", "c": "complied"}


def read_labels(path: Path) -> dict[str, str]:
    """Existing labels keyed by conversation id."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row["label"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, default=Path("data/multi_turn_gen"))
    parser.add_argument("--out", type=Path, default=Path("data/output_labels.csv"))
    args = parser.parse_args()

    conversations = load_dir(args.conversations)
    labels = read_labels(args.out)
    pending = [conv for conv in conversations if conv["id"] not in labels]
    for number, conv in enumerate(pending, start=1):
        last = conv["turns"][-1]
        print(f"\n[{number}/{len(pending)}] {conv['id']} ({conv['category']})")
        print(f"USER: {last['user']}\nREPLY: {last['assistant']}\n")
        answer = ""
        while answer not in CHOICES and answer != "q":
            answer = input("r = refused, p = partial, c = complied, q = quit: ").strip().lower()
        if answer == "q":
            break
        labels[conv["id"]] = CHOICES[answer]
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "label"])
            writer.writerows(labels.items())
    print(f"\nLabelled {len(labels)} of {len(conversations)}: {dict(Counter(labels.values()))}")


if __name__ == "__main__":
    main()
