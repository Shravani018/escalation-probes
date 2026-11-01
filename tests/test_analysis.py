"""PCA, probes, layer selection, metrics and judge parsing on synthetic activations."""

import numpy as np
import pandas as pd

from escalation_probes.judge import Judge, parse_verdict, render_conversation
from escalation_probes.metrics import compare_methods, detection_accuracy, earliest_flag, probe_turn_flags, summarise
from escalation_probes.pca import fit_reference
from escalation_probes.probe import example_accuracy, harmful_share, make_probe, pick_layers

CFG = {"probe": {"hidden_layer_sizes": [16, 8], "max_iter": 500, "random_state": 42}}


def blobs(n_per_class=300, dim=20, gap=4.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(0, 1, (n_per_class, dim)), rng.normal(0, 1, (n_per_class, dim)) + gap])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y


def test_reference_space_places_tokens_near_their_class_centroid():
    x, y = blobs()
    space = fit_reference(x, y, n_components=5, seed=0)
    benign = space.centroid_report(x[y == 0], 2)
    harmful = space.centroid_report(x[y == 1], 2)
    assert benign["share_nearer_harmful"] < 0.1
    assert harmful["share_nearer_harmful"] > 0.9
    assert space.variance_explained[-1] <= 1.0


def test_probes_separate_classes_and_report_example_accuracy():
    x, y = blobs()
    groups = np.repeat(np.arange(len(y) // 10), 10)
    for kind in ("mlp", "linear"):
        probe = make_probe(kind, CFG).fit(x, y)
        result = example_accuracy(probe, x, y, groups)
        assert result["token_accuracy"] > 0.95 and result["example_accuracy"] > 0.95


def test_harmful_share_tracks_class():
    x, y = blobs()
    probe = make_probe("linear", CFG).fit(x, y)
    assert harmful_share(probe, x[y == 0])[0] < 0.1
    assert harmful_share(probe, x[y == 1])[0] > 0.9


def test_pick_layers_alternate_is_not_adjacent():
    choice = pick_layers({8: 0.90, 12: 0.95, 16: 0.94, 20: 0.80, 24: 0.92})
    assert choice["selected"] == 12
    assert choice["alternate"] == 24


def test_pick_layers_with_few_candidates():
    assert pick_layers({4: 0.9, 8: 0.8})["alternate"] == 8


def test_earliest_flag_ignores_unscored_turns():
    assert earliest_flag({1: False, 2: None, 3: True, 4: True}) == 3
    assert earliest_flag({1: False, 2: None}) is None


def test_detection_accuracy():
    assert detection_accuracy([True, False, True], [True, False, False]) == 2 / 3


def test_compare_and_summarise():
    rows = []
    for rec_id, label, shares in [("e1", "escalation", [0.0, 0.2, 0.6, 0.8]), ("c1", "benign_control", [0.0, 0.1, 0.1, 0.0])]:
        for turn, share in enumerate(shares, start=1):
            rows.append(dict(model="m", probe_kind="mlp", layer_role="selected", record_id=rec_id, label=label,
                             condition="crescendo", turn=turn, k=turn, harmful_share=share))
        rows.append(dict(model="m", probe_kind="mlp", layer_role="selected", record_id=rec_id, label=label,
                         condition="crescendo", turn=4, k=1, harmful_share=0.99))  # k sweep row must be ignored
    flags = probe_turn_flags(pd.DataFrame(rows), conversation_share=0.5)
    judge = pd.DataFrame(
        [{"record_id": "e1", "turn": t, "flag": f} for t, f in [(1, False), (2, True), (3, True), (4, True)]]
        + [{"record_id": "c1", "turn": t, "flag": None} for t in range(1, 5)]
    )
    comparison = compare_methods(flags, judge).set_index("record_id")
    assert comparison.loc["e1", "probe_earliest"] == 3
    assert comparison.loc["e1", "judge_earliest"] == 2
    assert bool(comparison.loc["c1", "probe_final"]) is False
    assert comparison.loc["c1", "judge_unscored_turns"] == 4
    summary = summarise(comparison.reset_index())
    assert summary.loc[0, "probe_accuracy"] == 1.0
    assert summary.loc[0, "judge_scored"] == 1


def test_parse_verdict_handles_wrapped_and_invalid_replies():
    assert parse_verdict('Result: {"flag": true, "score": 72, "rationale": "drift"}')["score"] == 72.0
    assert parse_verdict("UNABLE") is None
    assert parse_verdict('{"flag": "yes", "score": 5}') is None
    assert parse_verdict("{not json}") is None


def test_judge_reviews_and_marks_unscored_turns():
    replies = iter(['{"flag": true, "score": 60, "rationale": "a"}', '{"flag": false, "score": 40, "rationale": "b"}', "UNABLE"])
    judge = Judge(lambda system, user: next(replies), "sys", "review")
    turns = [{"user": "u1", "assistant": "a1"}, {"user": "u2", "assistant": "a2"}]
    reviewed = judge.score_turn(turns, 1)
    assert (reviewed["status"], reviewed["flag"], reviewed["first_flag"]) == ("ok", False, True)
    unscored = judge.score_turn(turns, 2)
    assert unscored["status"] == "unparsed" and unscored["flag"] is None


def test_transcript_stops_at_requested_turn():
    turns = [{"user": "u1", "assistant": "a1"}, {"user": "u2", "assistant": "a2"}]
    assert "u2" not in render_conversation(turns, 1)
