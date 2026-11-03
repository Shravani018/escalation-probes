"""Compare probe and judge on the same conversations: final-turn accuracy and earliest flagged turn."""

import argparse
import logging

import pandas as pd

from escalation_probes.config import load_experiment, tables_dir
from escalation_probes.metrics import compare_methods, probe_turn_flags, summarise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    out = tables_dir(cfg)
    probe = pd.read_csv(out / f"probe_multi_turn_{args.model}.csv")
    judge = pd.read_json(out / "judge_turns.jsonl", lines=True)
    flags = probe_turn_flags(probe, cfg["decision"]["conversation_share"])
    comparison = compare_methods(flags, judge[["record_id", "turn", "flag"]])
    comparison.to_csv(out / f"comparison_{args.model}.csv", index=False)
    summary = summarise(comparison)
    summary.to_csv(out / f"summary_{args.model}.csv", index=False)
    log.info("Wrote comparison tables to %s", out)
    log.info("Summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
