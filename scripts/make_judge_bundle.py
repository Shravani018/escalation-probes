"""Zip the files the black-box judge needs, for upload to Colab or another machine."""

import argparse
import zipfile
from pathlib import Path

CORE = [
    "scripts/run_judge.py",
    "configs/experiment.yaml",
    "prompts/judge_v1.md",
    "prompts/judge_v1_review.md",
    "src/escalation_probes/__init__.py",
    "src/escalation_probes/config.py",
    "src/escalation_probes/data.py",
    "src/escalation_probes/judge.py",
]
DATA_PATTERNS = ["data/multi_turn_gen*/*.jsonl", "data/controls_gen*/*.jsonl"]
LABEL_PATTERN = "data/output_labels*.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="judge_bundle.zip")
    parser.add_argument("--with-labels", action="store_true", help="Also include data/output_labels*.csv.")
    args = parser.parse_args()

    root = Path(".")
    missing = [p for p in CORE if not (root / p).exists()]
    if missing:
        raise SystemExit(f"Run from the repo root. Missing: {missing}")
    files = list(CORE)
    for pattern in DATA_PATTERNS + ([LABEL_PATTERN] if args.with_labels else []):
        files += sorted(str(p).replace("\\", "/") for p in root.glob(pattern))
    if not any(f.startswith("data/multi_turn_gen") for f in files):
        raise SystemExit("No data/multi_turn_gen*/ conversations found. Run generate_replies.py first.")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.write(name, name)
    print(f"Wrote {args.out} with {len(files)} files:")
    for name in files:
        print(" ", name)


if __name__ == "__main__":
    main()
