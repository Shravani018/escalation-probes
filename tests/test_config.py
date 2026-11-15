"""A data tag points conversation paths at a per-model folder and leaves everything else alone."""

from pathlib import Path

from escalation_probes.config import load_experiment

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


def test_tag_redirects_conversation_paths_only():
    plain = load_experiment(CONFIG)
    tagged = load_experiment(CONFIG, "olmo-2-1b")
    assert tagged["paths"]["multi_turn"] == plain["paths"]["multi_turn"] + "_olmo-2-1b"
    assert tagged["paths"]["controls"] == plain["paths"]["controls"] + "_olmo-2-1b"
    assert tagged["paths"]["single_turn"] == plain["paths"]["single_turn"]


def test_empty_tag_keeps_defaults():
    assert load_experiment(CONFIG, "") == load_experiment(CONFIG)
