"""Splits are grouped by example and no conversation text reaches single-turn training data."""

import pytest

from escalation_probes.data import check_no_leakage, make_splits

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_splits_are_disjoint_and_complete(single_turn, conversations):
    ids = [rec["id"] for rec in single_turn]
    splits = make_splits(ids, FRACTIONS, seed=42)
    check_no_leakage(splits, single_turn, conversations)
    assert sorted(sum(splits.values(), [])) == sorted(ids)


def test_splits_are_deterministic():
    ids = [f"id-{i}" for i in range(50)]
    assert make_splits(ids, FRACTIONS, 7) == make_splits(ids, FRACTIONS, 7)


def test_overlapping_splits_are_rejected(single_turn, conversations):
    ids = [rec["id"] for rec in single_turn]
    splits = make_splits(ids, FRACTIONS, 42)
    splits["test"].append(splits["train"][0])
    with pytest.raises(AssertionError, match="shared"):
        check_no_leakage(splits, single_turn, conversations)


def test_conversation_text_in_single_turn_is_rejected(single_turn, conversations):
    splits = make_splits([rec["id"] for rec in single_turn], FRACTIONS, 42)
    conversations[0]["turns"][1]["user"] = single_turn[0]["prompt"].upper()
    with pytest.raises(AssertionError, match="reuses"):
        check_no_leakage(splits, single_turn, conversations)


def test_unknown_split_id_is_rejected(single_turn, conversations):
    splits = make_splits([rec["id"] for rec in single_turn], FRACTIONS, 42)
    splits["val"].append("not-a-real-id")
    with pytest.raises(AssertionError, match="absent"):
        check_no_leakage(splits, single_turn, conversations)
