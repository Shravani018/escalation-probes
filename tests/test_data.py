"""Condition construction, job lists and validation."""

import pytest

from conftest import make_conversation
from escalation_probes.data import build_messages, iter_jobs, validate_conversations, validate_single_turn


def test_crescendo_keeps_last_k_pairs_and_final_response():
    conv = make_conversation("c", "escalation", n_turns=4)
    messages = build_messages(conv, turn=4, k=2, condition="crescendo")
    assert [m["content"] for m in messages] == ["c user 3", "c assistant 3", "c user 4", "c assistant 4"]
    assert messages[-1]["role"] == "assistant"


def test_final_response_is_identical_across_k_and_conditions():
    conv = make_conversation("c", "escalation", n_turns=4)
    finals = {
        build_messages(conv, 4, k, cond)[-1]["content"]
        for cond in ("crescendo", "single_prompt", "attack_objective")
        for k in (1, 2, 3, 4)
    }
    assert finals == {"c assistant 4"}


def test_single_prompt_flattens_history_into_one_user_message():
    conv = make_conversation("c", "escalation", n_turns=3)
    messages = build_messages(conv, 3, 3, "single_prompt")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert "User: c user 1\nAssistant: c assistant 1" in messages[0]["content"]
    assert messages[0]["content"].endswith("User: c user 3")


def test_single_prompt_k1_matches_crescendo_k1():
    conv = make_conversation("c", "escalation", n_turns=3)
    assert build_messages(conv, 3, 1, "single_prompt") == build_messages(conv, 3, 1, "crescendo")


def test_attack_objective_uses_objective_only():
    conv = make_conversation("c", "escalation")
    assert build_messages(conv, 4, 1, "attack_objective")[0]["content"] == conv["objective"]


def test_invalid_turn_or_k_raises():
    conv = make_conversation("c", "escalation", n_turns=3)
    with pytest.raises(ValueError):
        build_messages(conv, 4, 1, "crescendo")
    with pytest.raises(ValueError):
        build_messages(conv, 2, 3, "crescendo")


def test_jobs_are_distinct_and_cover_full_history_per_turn():
    jobs = iter_jobs(make_conversation("c", "escalation", n_turns=4))
    keys = [(j["condition"], j["turn"], j["k"]) for j in jobs]
    assert len(keys) == len(set(keys))
    assert {("crescendo", t, t) for t in range(1, 5)} <= set(keys)
    assert {("crescendo", 4, k) for k in range(1, 5)} <= set(keys)
    assert ("attack_objective", 4, 1) in keys


def test_control_must_match_turn_count_and_category():
    escalation = make_conversation("e", "escalation", n_turns=4)
    control = make_conversation("c", "benign_control", n_turns=3, matched_to="e")
    with pytest.raises(ValueError, match="differs"):
        validate_conversations([escalation, control])
    control = make_conversation("c", "benign_control", n_turns=4, category="phishing_fraud", matched_to="e")
    with pytest.raises(ValueError, match="differs"):
        validate_conversations([escalation, control])


def test_validation_accepts_good_data_and_rejects_duplicates(single_turn, conversations):
    validate_single_turn(single_turn)
    validate_conversations(conversations)
    with pytest.raises(ValueError, match="duplicate"):
        validate_single_turn(single_turn + single_turn[:1])
