"""Prepare a deterministic 500-benign/500-harmful reference set from local sources."""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


URL = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|io|ly)(?:/\S*)?", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
CODE_BLOCK = re.compile(r"```.*?```", re.S)


def redact(text: str) -> str:
    """Replace common directly actionable identifiers before writing records."""
    text = CODE_BLOCK.sub("[REDACTED]", text)
    text = URL.sub("[LINK]", text)
    text = EMAIL.sub("[CREDENTIAL]", text)
    return IPV4.sub("[TARGET]", text)


def load_harmful(path: Path) -> list[dict]:
    """Load harmful prompt-response records from the local JSON source."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "id": f"cb-{index:04d}",
            "label": "harmful",
            "category": record.get("category", "unspecified"),
            "prompt": redact(record["prompt"]),
            "response": redact(record["output"]),
            "source": "circuit_breakers_train.json",
            "source_index": index,
        }
        for index, record in enumerate(records)
        if record.get("prompt", "").strip() and record.get("output", "").strip()
    ]


def load_benign(path: Path) -> list[dict]:
    """Load prompt and final assistant-response pairs from the local parquet source."""
    frame = pd.read_parquet(path, columns=["prompt", "messages"])
    records = []
    for index, row in frame.iterrows():
        messages = row["messages"]
        assistants = [item["content"] for item in messages if item["role"] == "assistant"]
        if not str(row["prompt"]).strip() or not assistants or not assistants[-1].strip():
            continue
        records.append(
            {
                "id": f"retain-{index:05d}",
                "label": "benign",
                "category": "retain",
                "prompt": redact(str(row["prompt"])),
                "response": redact(str(assistants[-1])),
                "source": path.name,
                "source_index": int(index),
            }
        )
    return records


def sample(records: list[dict], count: int, seed: int) -> list[dict]:
    """Select a reproducible sample and return it in stable id order."""
    if len(records) < count:
        raise ValueError(f"source has {len(records)} usable records; need {count}")
    selected = random.Random(seed).sample(records, count)
    return sorted(selected, key=lambda record: record["id"])


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records without printing their prompt or response text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmful", default="data/source/circuit_breakers_train.json")
    parser.add_argument("--benign", default="data/source/test_gen-00000-of-00001-3d4cd8309148a71f.parquet")
    parser.add_argument("--output", default="data/single_turn_reference")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    harmful = sample(load_harmful(Path(args.harmful)), args.count, args.seed)
    benign = sample(load_benign(Path(args.benign)), args.count, args.seed + 1)
    output = Path(args.output)
    write_jsonl(output / "harmful.jsonl", harmful)
    write_jsonl(output / "benign.jsonl", benign)
    print(f"wrote {len(benign)} benign and {len(harmful)} harmful records to {output}")


if __name__ == "__main__":
    main()
