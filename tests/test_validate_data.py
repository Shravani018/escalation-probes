"""Tests for the pre-run conversation data validator."""

from scripts.validate_data import check_groups, check_record


def conversation(record_id, label, matched_to=None):
    record = {
        "id": record_id,
        "label": label,
        "category": "cyber_intrusion",
        "objective": "redacted objective",
        "turns": [
            {"user": "safe topic", "assistant": "safe response"},
            {"user": "more detail", "assistant": "redacted response"},
        ],
    }
    if matched_to is not None:
        record["matched_to"] = matched_to
    return record


def test_validator_accepts_runtime_turn_schema():
    assert check_record(conversation("e1", "escalation")) == []


def test_validator_rejects_legacy_role_content_schema():
    record = conversation("e1", "escalation")
    record["turns"] = [{"role": "user", "content": "safe topic"}]
    assert "turns need user and assistant text" in check_record(record)[0]


def test_validator_checks_control_matching():
    escalation = conversation("e1", "escalation")
    control = conversation("c1", "benign_control", matched_to="e1")
    assert check_groups([escalation, control]) == []
