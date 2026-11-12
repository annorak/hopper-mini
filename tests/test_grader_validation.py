import asyncio
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from typer.testing import CliRunner
from typesafe_sdk import NoulAnswer, SystemOneResponse, TypeSafeBadRequestError, Usage

from hopper.config import Settings
from hopper.graders import (
    Catalog,
    Validation,
    definition_fingerprint,
    read_definition,
)
from hopper.graders import jev as jev_module
from hopper.graders.jev import canonical_state, hash_state
from hopper.graders.validation import validate_graders
from hopper.ledger import Ledger, SpendCapExceeded


@pytest.fixture
def environment(tmp_path, monkeypatch):
    settings = Settings(tmp_path / "store", typesafe_api_key="test-key")
    Ledger(settings).initialize()
    catalog = Catalog(settings.data_dir)
    grader = catalog.create_draft(
        {
            "name": "grounded",
            "kind": "jev",
            "applies_to": "rfi-agent",
            "question": "Every claim is supported.",
            "fields": ["final_answer", "retrieved_text"],
            "yes_threshold": 0.7,
            "no_threshold": 0.3,
            "model": "jev-1.14.0",
            "origin": "human",
        }
    )
    sdk = SimpleNamespace(system_one=AsyncMock(), aclose=AsyncMock())
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(jev_module, "AsyncTypeSafeClient", factory)
    return settings, catalog, grader, tmp_path / "labels", sdk, factory


def response(probability, grader):
    return SystemOneResponse(
        model=grader.model,
        usage=Usage(input_tokens=100),
        answers={f"{grader.name}_v{grader.version}": NoulAnswer(noul=probability)},
    )


def save_labels(directory, grader, answers):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, answer in enumerate(answers):
        state = {"final_answer": {"text": f"Saved answer {index}"}, "retrieved_text": "Evidence"}
        records.append(
            {
                "grader": grader.name,
                "grader_version": grader.version,
                "question": grader.question,
                "fields": list(grader.fields),
                "applies_to": grader.applies_to,
                "trace_id": f"trace-{index}",
                "task_id": f"task-{index}",
                "checker_passed": index % 2 == 0,
                "labeled_at": "2026-11-11T00:00:00+00:00",
                "state": state,
                "input_hash": hash_state(state),
                "label": answer,
            }
        )
    path = directory / f"{grader.name}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return records


def run_validation(environment):
    settings, _, grader, labels, _, _ = environment
    return asyncio.run(validate_graders(settings, [grader], labels, rps=1_000_000))


def test_validation_uses_saved_states_without_source_store_and_sweeps_locally(environment):
    settings, catalog, grader, labels, sdk, _ = environment
    records = save_labels(labels, grader, ["no", "no", "yes", "no", "yes", "yes"])
    sdk.system_one.side_effect = [response(p, grader) for p in (0.1, 0.3, 0.3, 0.5, 0.7, 0.9)]
    assert list((settings.data_dir / "spans").iterdir()) == []
    assert list((settings.data_dir / "corpus").iterdir()) == []
    [(validated, report)] = run_validation(environment)
    assert report.labels == 6
    assert report.agreement == pytest.approx(4 / 6)
    assert report.precision_no == pytest.approx(2 / 3)
    assert report.recall_no == pytest.approx(2 / 3)
    assert [(row.no_threshold, row.yes_threshold) for row in report.threshold_sweep] == [
        (0.1, 0.9),
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
    ]
    assert [row.agreement for row in report.threshold_sweep] == pytest.approx(
        [2 / 6, 2 / 6, 4 / 6, 4 / 6]
    )
    assert report.threshold_sweep[0].precision_no == 1
    assert report.threshold_sweep[0].recall_no == pytest.approx(1 / 3)
    assert [call.kwargs["state"] for call in sdk.system_one.call_args_list] == [
        record["state"] for record in records
    ]
    assert sdk.system_one.await_count == len(records)
    assert report.definition_hash == definition_fingerprint(grader)
    evidence = sorted(
        ({key: record[key] for key in ("state", "input_hash", "label")} for record in records),
        key=lambda item: item["input_hash"],
    )
    assert (
        report.labels_hash
        == hashlib.sha256(canonical_state({"labels": evidence}).encode()).hexdigest()
    )
    assert validated.validation == report
    assert catalog.list_graders() == [validated]
    sdk.aclose.assert_awaited_once()
    ledger = [
        json.loads(line) for line in (settings.data_dir / "ledger.jsonl").read_text().splitlines()
    ]
    assert len(ledger) == 6
    assert all(row["stage"] == "grading" and row["state"] == "settled" for row in ledger)
    with pytest.raises(ValueError, match="50 labels"):
        catalog.promote(grader.name, grader.version, run_code_tests=Mock())


@pytest.mark.parametrize(
    ("human", "probability", "agreement", "precision", "recall"),
    [
        ("yes", 1.0, 1.0, None, None),
        ("no", 1.0, 0.0, None, 0.0),
        ("yes", 0.0, 0.0, 0.0, None),
        ("no", 0.5, 0.0, None, 0.0),
    ],
)
def test_undefined_denominators_are_null_and_unsure_is_disagreement(
    environment, human, probability, agreement, precision, recall
):
    _, _, grader, labels, sdk, _ = environment
    save_labels(labels, grader, [human])
    sdk.system_one.return_value = response(probability, grader)
    [(_, report)] = run_validation(environment)
    assert (report.agreement, report.precision_no, report.recall_no) == (
        agreement,
        precision,
        recall,
    )


def test_custom_threshold_pair_is_inclusive_and_is_added_exactly(environment):
    settings, catalog, grader, labels, sdk, _ = environment
    path = catalog.directory / grader.name / "v1.yaml"
    document = yaml.safe_load(path.read_text())
    document.update(no_threshold=0.25, yes_threshold=0.75)
    path.write_text(yaml.safe_dump(document))
    grader = read_definition(path)
    save_labels(labels, grader, ["no", "yes", "no"])
    sdk.system_one.side_effect = [response(p, grader) for p in (0.25, 0.75, 0.5)]
    [(_, report)] = asyncio.run(validate_graders(settings, [grader], labels, rps=1_000_000))
    assert len(report.threshold_sweep) == 5
    current = report.threshold_sweep[-1]
    assert (current.no_threshold, current.yes_threshold) == (0.25, 0.75)
    assert (report.agreement, report.precision_no, report.recall_no) == (2 / 3, 1, 1 / 2)
    assert current.agreement == report.agreement
    assert sdk.system_one.await_count == 3


def test_zero_labels_persist_unavailable_report_without_client_or_key(environment):
    settings, catalog, grader, labels, sdk, factory = environment
    settings = replace(settings, typesafe_api_key=None)
    [(_, report)] = asyncio.run(validate_graders(settings, [grader], labels))
    assert report.labels == 0
    assert (report.agreement, report.precision_no, report.recall_no) == (None, None, None)
    assert all(
        (row.agreement, row.precision_no, row.recall_no) == (None, None, None)
        for row in report.threshold_sweep
    )
    assert (
        report.labels_hash == hashlib.sha256(canonical_state({"labels": []}).encode()).hexdigest()
    )
    assert catalog.list_graders()[0].validation == report
    factory.assert_not_called()
    sdk.system_one.assert_not_awaited()


def test_request_rate_limit_is_shared_across_graders(environment, monkeypatch):
    settings, catalog, grader, labels, sdk, factory = environment
    definition = yaml.safe_load((catalog.directory / grader.name / "v1.yaml").read_text())
    for field in ("version", "status", "validation"):
        definition.pop(field)
    second = catalog.create_draft({**definition, "name": "supported"})
    for candidate in (grader, second):
        save_labels(labels, candidate, ["yes"])
    clock = [0.0]
    dispatched = []

    async def advance(seconds):
        clock[0] += seconds

    async def judge(**request):
        dispatched.append(clock[0])
        candidate = grader if f"{grader.name}_v1" in request["questions"] else second
        return response(0.9, candidate)

    monkeypatch.setattr(jev_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(jev_module.asyncio, "sleep", advance)
    sdk.system_one.side_effect = judge
    reports = asyncio.run(validate_graders(settings, [grader, second], labels, rps=0.1))
    assert dispatched == [0.0, 10.0]
    assert [report.labels for _, report in reports] == [1, 1]
    factory.assert_called_once()
    sdk.aclose.assert_awaited_once()


@pytest.mark.parametrize("failure", ["request", "token_retry", "budget"])
def test_partial_failure_does_not_publish_validation(environment, failure):
    settings, catalog, grader, labels, sdk, _ = environment
    [(_, previous)] = run_validation(environment)
    save_labels(labels, grader, ["yes", "yes"])
    path = catalog.directory / grader.name / "v1.yaml"
    before = path.read_bytes()
    if failure == "request":
        sdk.system_one.side_effect = [response(0.9, grader), RuntimeError("request failed")]
        expected, message, count = RuntimeError, "request failed", 2
    elif failure == "token_retry":
        sdk.system_one.side_effect = [
            response(0.9, grader),
            TypeSafeBadRequestError(400, {"detail": "max_tokens_exceeded"}, {}),
            response(0.9, grader),
        ]
        expected, message, count = ValueError, "input changed", 3
    else:
        settings = replace(settings, spend_cap_usd=Decimal("0"))
        expected, message, count = SpendCapExceeded, "cap", 0
    with pytest.raises(expected, match=message):
        asyncio.run(validate_graders(settings, [grader], labels, rps=1_000_000))
    assert path.read_bytes() == before
    assert catalog.list_graders()[0].validation == previous
    assert sdk.system_one.await_count == count
    sdk.aclose.assert_awaited_once()


def test_fifty_judged_labels_pass_the_exact_promotion_gate(environment):
    _, catalog, grader, labels, sdk, _ = environment
    save_labels(labels, grader, ["no"] * 20 + ["yes"] * 30)
    sdk.system_one.side_effect = [response(p, grader) for p in [0.1] * 20 + [0.9] * 30]
    [(_, report)] = run_validation(environment)
    assert report.labels == 50 and report.agreement == 1
    promoted = catalog.promote(grader.name, grader.version, run_code_tests=Mock())
    assert promoted.status == "active"
    assert promoted.validation == report


def test_report_preserves_status_changed_during_validation(environment):
    _, catalog, grader, _, _, _ = environment
    [(_, report)] = run_validation(environment)
    catalog.retire(grader.name, grader.version)
    saved = catalog.save_validation(grader, report)
    assert saved.status == "retired"
    assert saved.validation == report


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "A different question."),
        ("model", "jev-1.15.0"),
        ("yes_threshold", 0.8),
        ("no_threshold", 0.2),
        ("fields", ["final_answer"]),
        ("applies_to", "a-different-service"),
        ("origin", "discovered"),
    ],
)
def test_changed_definition_rejects_stale_save_and_promotion(environment, field, value):
    _, catalog, grader, labels, sdk, _ = environment
    save_labels(labels, grader, ["yes"] * 50)
    sdk.system_one.return_value = response(0.9, grader)
    [(_, report)] = run_validation(environment)
    path = catalog.directory / grader.name / "v1.yaml"
    document = yaml.safe_load(path.read_text())
    document[field] = value
    path.write_text(yaml.safe_dump(document))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed during validation"):
        catalog.save_validation(grader, report)
    with pytest.raises(ValueError, match="full report bound"):
        catalog.promote(grader.name, grader.version, run_code_tests=Mock())
    assert path.read_bytes() == before


@pytest.mark.parametrize("labels", [0, 50])
def test_legacy_report_is_readable_but_cannot_authorize_promotion(environment, labels):
    _, catalog, grader, _, _, _ = environment
    path = catalog.directory / grader.name / "v1.yaml"
    document = yaml.safe_load(path.read_text())
    document["validation"] = {"labels": labels, "agreement": 1.0}
    path.write_text(yaml.safe_dump(document))
    assert read_definition(path).validation == Validation(labels, 1.0)
    with pytest.raises(ValueError, match="50 labels|full report"):
        catalog.promote(grader.name, grader.version, run_code_tests=Mock())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("labels", True),
        ("labels", -1),
        ("labels", 1),
        ("agreement", 0),
        ("precision_no", 0),
        ("recall_no", False),
        ("validated_at", None),
        ("validated_at", "2026-11-11T00:00:00"),
        ("definition_hash", "bad"),
        ("labels_hash", None),
        ("threshold_sweep", None),
        ("threshold_sweep", []),
        ("unknown", 0),
    ],
)
def test_malformed_full_reports_are_rejected(environment, key, value):
    _, catalog, grader, _, _, _ = environment
    [(_, report)] = run_validation(environment)
    path = catalog.directory / grader.name / "v1.yaml"
    document = yaml.safe_load(path.read_text())
    document["validation"][key] = value
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        read_definition(path)
    assert report.labels == 0


@pytest.mark.parametrize("mutation", ["missing", "unknown", "duplicate", "metric", "threshold"])
def test_malformed_threshold_rows_are_rejected(environment, mutation):
    _, catalog, grader, _, _, _ = environment
    run_validation(environment)
    path = catalog.directory / grader.name / "v1.yaml"
    document = yaml.safe_load(path.read_text())
    sweep = document["validation"]["threshold_sweep"]
    if mutation == "missing":
        del sweep[0]["recall_no"]
    elif mutation == "unknown":
        sweep[0]["unknown"] = 1
    elif mutation == "duplicate":
        sweep.append(sweep[0])
    elif mutation == "metric":
        sweep[0]["agreement"] = 1
    else:
        sweep[0]["no_threshold"] = sweep[0]["yes_threshold"]
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        read_definition(path)


def test_save_rejects_incomplete_sweep_and_inconsistent_current_metrics(environment):
    _, catalog, grader, labels, sdk, _ = environment
    save_labels(labels, grader, ["yes"])
    sdk.system_one.return_value = response(0.9, grader)
    [(_, report)] = run_validation(environment)
    with pytest.raises(ValueError, match="required threshold pairs"):
        catalog.save_validation(
            grader, replace(report, threshold_sweep=report.threshold_sweep[:-1])
        )
    with pytest.raises(ValueError, match="configured threshold row"):
        catalog.save_validation(grader, replace(report, agreement=0.0))


def test_report_labels_hash_ignores_order_and_provenance_but_binds_human_answers(environment):
    _, _, grader, labels, sdk, _ = environment
    records = save_labels(labels, grader, ["yes", "no"])
    sdk.system_one.return_value = response(0.9, grader)
    [(_, original)] = run_validation(environment)
    for record in records:
        record["trace_id"] += "-rebuilt"
    path = labels / f"{grader.name}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in reversed(records)))
    [(_, rebuilt)] = run_validation(environment)
    assert rebuilt.labels_hash == original.labels_hash
    records[0]["label"] = "no"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    [(_, relabeled)] = run_validation(environment)
    assert relabeled.labels_hash != original.labels_hash


def test_corrupt_saved_state_fails_before_client_or_report_publication(environment):
    _, catalog, grader, labels, sdk, factory = environment
    records = save_labels(labels, grader, ["yes"])
    records[0]["state"]["retrieved_text"] = "Changed since the human answer"
    (labels / f"{grader.name}.jsonl").write_text(json.dumps(records[0]) + "\n")
    path = catalog.directory / grader.name / "v1.yaml"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="hash"):
        run_validation(environment)
    assert path.read_bytes() == before
    factory.assert_not_called()
    sdk.system_one.assert_not_awaited()


def test_cli_all_zero_labels_prints_unavailable_metrics_and_persists_yaml(environment, monkeypatch):
    from hopper.cli import app

    settings, catalog, grader, labels, sdk, factory = environment
    catalog.create_draft(
        {
            "name": "cites_sources",
            "kind": "code",
            "applies_to": "rfi-agent",
            "fields": ["final_answer"],
            "origin": "human",
            "function": "hopper.graders.code:cites_sources",
        }
    )
    monkeypatch.setenv("HOPPER_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    result = CliRunner().invoke(app, ["graders", "validate", "--all", "--labels-dir", str(labels)])
    assert result.exit_code == 0, result.output
    assert "grader\tversion\tlabels\tagreement\tprecision_no\trecall_no" in result.output
    assert "grounded\t1\t0\tn/a\tn/a\tn/a" in result.output
    assert "Code graders use unit tests and are excluded" in result.output
    assert "No human labels recorded" in result.output
    assert "0.3\t0.7\tn/a\tn/a\tn/a" in result.output
    report = read_definition(catalog.directory / grader.name / "v1.yaml").validation
    assert report.labels == 0 and report.agreement is None
    assert report.definition_hash == definition_fingerprint(grader)
    factory.assert_not_called()
    sdk.system_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "Specify a grader name"),
        (["grounded", "--all"], "--all cannot be combined"),
        (["--all", "--version", "1"], "--all cannot be combined"),
        (["--version", "1"], "Specify a grader name"),
        (["missing"], "Unknown grader or version"),
        (["grounded", "--version", "99"], "Unknown grader or version"),
        (["grounded", "--usd-per-hour", "bad"], "USD amount"),
        (["grounded", "--usd-per-hour", "-1"], "USD amount"),
        (["grounded", "--rps", "0"], "Invalid value"),
    ],
)
def test_cli_invalid_validation_options_fail_before_any_judgment(
    environment, monkeypatch, arguments, message
):
    from hopper.cli import app

    settings, catalog, grader, labels, sdk, factory = environment
    monkeypatch.setenv("HOPPER_DATA_DIR", str(settings.data_dir))
    path = catalog.directory / grader.name / "v1.yaml"
    before = path.read_bytes()
    result = CliRunner().invoke(
        app, ["graders", "validate", *arguments, "--labels-dir", str(labels)]
    )
    assert result.exit_code != 0
    assert message in result.output
    assert path.read_bytes() == before
    factory.assert_not_called()
    sdk.system_one.assert_not_awaited()
