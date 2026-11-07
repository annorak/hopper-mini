import json
from collections import Counter
from copy import deepcopy
from decimal import Decimal

import pytest

from hopper.corpus import load_corpus
from hopper.tasks import (
    DEFAULT_DATASET,
    load_tasks,
    make_plans,
    selfcheck_tasks,
    validate_task,
    validate_tasks,
)


@pytest.fixture
def corpus():
    return load_corpus()


def draft(plan):
    key = plan["answer_key"]
    text = " ".join(key["required_phrases"] + key.get("conflict_documents", []))
    gold = {"text": text or "The requested result is provided.",
            "citations": list(key["required_citations"]),
            "needs_engineer": key["needs_engineer"]}
    if plan["mode"] == "calculation":
        gold["value"] = key["value"]
        gold["text"] = f"The requested result is {key['value']}."
    return {**{field: value for field, value in deepcopy(plan).items()
               if field not in {"brief", "sources"}},
            "question": plan["brief"], "gold_answer": gold,
            "generation": {"model": "claude-sonnet-5-5", "response_id": "msg_test",
                           "response_sha256": "a" * 64, "stage": "test_fixture"}}


def test_fixed_plans_cover_modes_cases_and_avoid_ambiguous_sections(corpus):
    plans = make_plans(corpus)
    assert plans == make_plans(corpus)
    assert len(plans) == 150
    assert [plan["id"] for plan in plans] == [f"RFI-{number:03d}" for number in range(1, 151)]
    assert set(Counter(plan["mode"] for plan in plans).values()) == {25}
    assert len({plan["brief"] for plan in plans}) == 150
    lookups = [plan for plan in plans if plan["mode"] == "lookup"]
    assert len({plan["case_id"].split("/")[0] for plan in lookups}) == 20
    for mode, cases in (("conflict", "conflicts"), ("missed_addendum", "changes"),
                        ("needs_engineer", "gaps"), ("calculation", "calculations")):
        assert {plan["case_id"] for plan in plans if plan["mode"] == mode} == {
            case["id"] for case in corpus.cases[cases]}
    blocked = {case["section"] for case in corpus.cases["changes"] + corpus.cases["gaps"]}
    blocked.update(case["spec"] for case in corpus.cases["conflicts"])
    for plan in plans:
        assert set(plan) == {"id", "mode", "case_id", "corpus_version", "answer_key",
                             "brief", "sources"}
        if plan["mode"] in {"lookup", "multi_section"}:
            assert not blocked.intersection(plan["answer_key"]["required_citations"])
        if plan["mode"] == "multi_section":
            assert len({reference.split("/")[0]
                        for reference in plan["answer_key"]["required_citations"]}) == 1
        for phrase in plan["answer_key"]["required_phrases"]:
            assert any(" ".join(phrase.split()) in " ".join(text.split())
                       for text in plan["sources"].values())


def test_all_gold_answers_validate_but_canonical_key_is_independent(corpus):
    plans = make_plans(corpus)
    tasks = [draft(plan) for plan in plans]
    report = validate_tasks(tasks, corpus)
    assert report["valid"] == 150
    assert report["error_count"] == 0
    assert report["total"] == 150
    assert selfcheck_tasks(tasks) == {"valid": True, "passing": 150, "failing": 0,
                                     "errors": [], "failed_criteria": {}}
    tasks[0]["answer_key"]["required_phrases"] = ["invented but self-consistent"]
    tasks[0]["gold_answer"]["text"] = "invented but self-consistent"
    assert selfcheck_tasks(tasks)["passing"] == 150
    assert "answer_key does not match the canonical plan" in validate_task(
        tasks[0], corpus, plans[0])


def test_committed_dataset_validates_and_every_gold_answer_passes(corpus):
    tasks = load_tasks(DEFAULT_DATASET)
    validation = validate_tasks(tasks, corpus)
    assert validation["valid"] == 150
    assert validation["error_count"] == 0
    selfcheck = selfcheck_tasks(tasks)
    assert selfcheck["passing"] == 150
    assert selfcheck["failing"] == 0


@pytest.mark.parametrize("field,value", [("id", "RFI-999"), ("mode", "other"),
                                       ("case_id", "unknown"), ("corpus_version", "stale"),
                                       ("question", "  "), ("question", 123),
                                       ("answer_key", None), ("gold_answer", []),
                                       ("generation", None)])
def test_task_boundary_rejects_bad_types_and_identity(corpus, field, value):
    plan = make_plans(corpus)[0]
    task = draft(plan)
    task[field] = value
    assert validate_task(task, corpus, plan)


@pytest.mark.parametrize("key,value", [("required_citations", [[]]),
                                     ("required_phrases", 42), ("needs_engineer", 1),
                                     ("mode", []), ("required_citations", ["SPEC-99/1.1"])])
def test_malformed_keys_report_errors_without_raising(corpus, key, value):
    plan = make_plans(corpus)[0]
    task = draft(plan)
    task["answer_key"][key] = value
    assert validate_task(task, corpus, plan)


@pytest.mark.parametrize("field,value", [("text", None), ("citations", "SPEC-01/1.1"),
                                       ("citations", [None]), ("citations", ["UNKNOWN"]),
                                       ("needs_engineer", "false"), ("value", True),
                                       ("value", float("nan"))])
def test_malformed_gold_reports_errors_without_raising(corpus, field, value):
    plan = make_plans(corpus)[0]
    task = draft(plan)
    task["gold_answer"][field] = value
    assert validate_task(task, corpus, plan)


@pytest.mark.parametrize("field,value", [("model", "claude-haiku-5-5"),
                                       ("response_id", ""), ("stage", []),
                                       ("response_sha256", "not-a-hash")])
def test_provenance_shape_is_required_and_extra_receipt_fields_are_allowed(corpus, field, value):
    plan = make_plans(corpus)[0]
    task = draft(plan)
    task["generation"]["usage"] = {"input_tokens": 10}
    assert not validate_task(task, corpus, plan)
    task["generation"][field] = value
    assert validate_task(task, corpus, plan)


def test_dataset_rejects_missing_modes_duplicate_ids_and_normalized_questions(corpus):
    tasks = [draft(plan) for plan in make_plans(corpus)]
    omitted = validate_tasks(tasks[:125], corpus)
    assert omitted["counts"]["calculation"] == 0
    assert any("Expected 25 calculation" in error for error in omitted["errors"])
    tasks[1]["question"] = "  " + tasks[0]["question"].upper().replace(" ", "\n ")
    tasks[2]["id"] = tasks[0]["id"]
    errors = validate_tasks(tasks, corpus)["errors"]
    assert any("IDs must be unique" in error for error in errors)
    assert any("Questions must be unique" in error for error in errors)
    assert validate_tasks([None, {"id": [], "mode": []}], corpus)["error_count"]


def test_numeric_keys_derive_from_quoted_tokens_and_detect_metadata_drift(corpus):
    plans = [plan for plan in make_plans(corpus) if plan["mode"] == "calculation"]
    results = {plan["case_id"]: plan["answer_key"]["value"] for plan in plans}
    assert results["CALC-02"] == 518400
    assert results["CALC-03"] == 72
    assert results["CALC-04"] == 3000
    assert results["CALC-05"] == 20.4
    for case, plan in zip(corpus.cases["calculations"], plans, strict=True):
        assert isinstance(plan["answer_key"]["value"], float)
        assert Decimal(str(plan["answer_key"]["tolerance"])) >= 0
        assert set(plan["answer_key"]["required_citations"]) == {
            item["citation"] for item in case["inputs"]}
    changed = deepcopy(corpus)
    changed.cases["calculations"][0]["inputs"][0]["value"] = 999999
    with pytest.raises(ValueError, match="differs from its source quote"):
        make_plans(changed)


def test_numeric_sentence_periods_and_unit_exponents_are_not_extra_values(corpus):
    cases = corpus.cases["calculations"]
    assert any(item["quote"] == "The simultaneous duty pump count is 2."
               for case in cases for item in case["inputs"])
    assert any(item["quote"].endswith("96 m3.") for case in cases for item in case["inputs"])
    assert len([plan for plan in make_plans(corpus) if plan["mode"] == "calculation"]) == 25


@pytest.mark.parametrize("mode,field", [("missed_addendum", "addendum_citation"),
                                      ("conflict", "precedence_citation"),
                                      ("conflict", "conflict_documents")])
def test_malformed_nested_mode_fields_report_errors(corpus, mode, field):
    plan = next(plan for plan in make_plans(corpus) if plan["mode"] == mode)
    task = draft(plan)
    task["answer_key"][field] = {"bad": []}
    assert validate_task(task, corpus, plan)


def test_source_quote_drift_is_rejected(corpus):
    changed = deepcopy(corpus)
    changed.cases["changes"][0]["after"] = "An invented source requirement."
    with pytest.raises(ValueError, match="Source quote"):
        make_plans(changed)


def test_jsonl_loader_reports_record_shape_and_nonfinite_values(tmp_path):
    path = tmp_path / "rfis.jsonl"
    path.write_text(json.dumps({"id": "RFI-001"}) + "\n\n", encoding="utf-8")
    assert load_tasks(path) == [{"id": "RFI-001"}]
    for content in ('[]\n', '{"value": NaN}\n', '{broken\n'):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="rfis.jsonl:1"):
            load_tasks(path)


def test_selfcheck_reports_failed_criteria(corpus):
    task = draft(make_plans(corpus)[0])
    task["gold_answer"]["citations"] = []
    report = selfcheck_tasks([task])
    assert report["passing"] == 0
    assert report["failing"] == 1
    assert report["errors"] == [{"id": task["id"], "failed_criteria": ["citations"]}]
    assert report["failed_criteria"] == {"citations": 1}
