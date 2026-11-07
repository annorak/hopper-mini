from copy import deepcopy

import pytest

from hopper.checks import check_answer


def key(mode="lookup", **fields):
    return {"mode": mode, "required_citations": ["SPEC-07/2.1"],
            "required_phrases": ["butterfly valve"], "needs_engineer": False, **fields}


def answer(**fields):
    return {"text": "Use a butterfly valve.", "citations": ["SPEC-07/2.1"],
            "needs_engineer": False, **fields}


def test_lookup_criteria_and_normalized_phrase_matching():
    result = check_answer(key(), answer(text="Use a BUTTERFLY\n  valve."))
    assert result == {"pass": True,
                      "criteria": {"citations": True, "phrases": True, "escalation": True}}
    assert not check_answer(key(), answer(citations=["SPEC-07/2.10"]))["criteria"]["citations"]
    assert not check_answer(key(), answer(text="Use a gate valve."))["criteria"]["phrases"]
    assert not check_answer(key(), answer(needs_engineer=True))["criteria"]["escalation"]


def test_multi_section_requires_every_exact_citation_and_phrase():
    expected = key("multi_section", required_citations=["SPEC-07/2.1", "SPEC-08/1.1"],
                   required_phrases=["butterfly valve", "stainless steel"])
    good = answer(text="Use a butterfly valve with stainless steel.",
                  citations=["SPEC-07/2.1", "SPEC-08/1.1"])
    assert check_answer(expected, good)["pass"]
    assert not check_answer(expected, answer())["criteria"]["citations"]
    assert not check_answer(expected, answer(citations=good["citations"]))["criteria"]["phrases"]


def test_addendum_has_an_independent_exact_item_criterion():
    expected = key("missed_addendum", addendum_citation="ADD-1/1")
    good = answer(citations=["SPEC-07/2.1", "ADD-1/1"])
    assert check_answer(expected, good)["pass"]
    result = check_answer(expected, answer())
    assert result["criteria"] == {"citations": True, "phrases": True,
                                  "addendum_cited": False, "escalation": True}
    assert not check_answer(expected, answer(citations=["SPEC-07/2.1", "ADD-1/10"]))["pass"]


def test_conflict_names_must_be_in_text_and_have_id_boundaries():
    expected = key("conflict", conflict_documents=["SPEC-07", "M-101"],
                   precedence_citation="SPEC-01/1.2")
    good = answer(text="SPEC-07 and M-101 differ. Use a butterfly valve.",
                  citations=["SPEC-07/2.1", "M-101/1", "SPEC-01/1.2"])
    assert check_answer(expected, good)["pass"]
    for text in ("Use a butterfly valve.", "SPEC-070 and M-101 use a butterfly valve.",
                 "SPEC-07 and M-1010 use a butterfly valve.",
                 "X-SPEC-07 and M-101 use a butterfly valve."):
        result = check_answer(expected, {**good, "text": text})
        assert not result["criteria"]["both_sides_named"]
        assert result["criteria"]["citations"]
        assert result["criteria"]["precedence_cited"]
    result = check_answer(expected, {**good, "citations": ["SPEC-07/2.1", "M-101/1"]})
    assert result["criteria"]["both_sides_named"]
    assert not result["criteria"]["precedence_cited"]


def test_engineer_has_no_citation_criterion_and_requires_a_real_boolean():
    expected = key("needs_engineer", required_citations=[], required_phrases=[],
                   needs_engineer=True)
    assert check_answer(expected, answer(needs_engineer=True)) == {
        "pass": True, "criteria": {"escalation": True}}
    assert not check_answer(expected, answer(needs_engineer="true"))["pass"]
    assert not check_answer(expected, answer(needs_engineer=1))["pass"]


@pytest.mark.parametrize("mode", ["lookup", "multi_section", "missed_addendum", "conflict",
                                 "needs_engineer", "calculation"])
def test_every_mode_checks_escalation(mode):
    expected = key(mode, needs_engineer=mode == "needs_engineer")
    result = check_answer(expected, answer(needs_engineer=mode != "needs_engineer"))
    assert result["criteria"]["escalation"] is False


@pytest.mark.parametrize("value,passes", [(0.3, True), (0.1, True), (0.30000000001, False),
                                         (0.09999999999, False), (None, False),
                                         (True, False), ("0.2", False),
                                         (float("nan"), False), (float("inf"), False)])
def test_calculation_absolute_tolerance_is_inclusive(value, passes):
    expected = key("calculation", required_phrases=[], value=0.2, tolerance=0.1)
    result = check_answer(expected, answer(value=value))
    assert result["criteria"]["value_in_tolerance"] is passes
    assert set(result["criteria"]) == {"citations", "escalation", "value_in_tolerance"}


def test_missing_value_and_nonfinite_key_fail_without_mutation():
    expected = key("calculation", value=12.5, tolerance=0)
    submitted = answer()
    before = deepcopy((expected, submitted))
    assert not check_answer(expected, submitted)["criteria"]["value_in_tolerance"]
    assert (expected, submitted) == before
    for value in (float("nan"), float("inf"), True):
        assert not check_answer({**expected, "value": value}, answer(value=12.5))["pass"]


@pytest.mark.parametrize("mode,field", [("missed_addendum", "addendum_citation"),
                                      ("conflict", "precedence_citation")])
def test_malformed_mode_citations_fail_without_raising(mode, field):
    assert not check_answer(key(mode, **{field: []}), answer())["pass"]
