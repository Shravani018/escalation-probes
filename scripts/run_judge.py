"""Run the frozen black-box judge on every turn of every conversation."""

import argparse
import json
from pathlib import Path

from escalation_probes.config import load_experiment, tables_dir
from escalation_probes.data import load_dataset
from escalation_probes.judge import Judge, prompt_hash


def anthropic_backend(model: str, max_tokens: int):
    """Return a complete(system, user) callable backed by the Anthropic API (needs `pip install anthropic`)."""
    import anthropic

    client = anthropic.Anthropic()

    def complete(system: str, user: str) -> str:
        reply = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in reply.content if block.type == "text")

    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    settings = cfg["judge"]
    if not settings["model"]:
        raise SystemExit("Set judge.model in configs/experiment.yaml before running the judge.")
    digest = prompt_hash(settings["prompt_file"], settings["review_prompt_file"])
    if settings["prompt_sha256"] and settings["prompt_sha256"] != digest:
        raise SystemExit("Judge prompt changed after it was frozen. Restore it or start a new prompt version.")

    judge = Judge(
        anthropic_backend(settings["model"], settings["max_tokens"]),
        Path(settings["prompt_file"]).read_text(encoding="utf-8"),
        Path(settings["review_prompt_file"]).read_text(encoding="utf-8"),
    )
    _, conversations = load_dataset(cfg)
    path = tables_dir(cfg) / "judge_turns.jsonl"
    done = {(r["record_id"], r["turn"]) for r in map(json.loads, path.open())} if path.exists() else set()
    with path.open("a", encoding="utf-8") as handle:
        for conv in conversations:
            for turn in range(1, len(conv["turns"]) + 1):
                if (conv["id"], turn) in done:
                    continue
                row = judge.score_turn(conv["turns"], turn)
                row["flag"] = None if row["score"] is None else row["score"] >= settings["flag_score_threshold"]
                row.update(record_id=conv["id"], label=conv["label"], judge_model=settings["model"], prompt_sha256=digest)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
    print(f"judge turns in {path}; prompt sha256 {digest}")
    print("Freeze the prompt by setting judge.prompt_sha256 to this value.")


if __name__ == "__main__":
    main()
