import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_grader_worker import ANSWER, MODEL, make_trace, publish
from typer.testing import CliRunner

from hopper.cli import app
from hopper.config import Settings
from hopper.graders import Catalog, JevGrader
from hopper.graders.jev import hash_state
from hopper.labels import Label, _append_label, label_traces, read_labels


def grader(name="answered", fields=("rfi_question",)):
    return JevGrader(
        name=name,
        version=1,
        status="draft",
        applies_to="rfi-agent",
        fields=fields,
        origin="human",
        question=f"Does {name} satisfy its criterion?",
        model=MODEL,
        yes_threshold=0.7,
        no_threshold=0.3,
    )


def trace(trace_id, *, question=None, evidence=None, answer=None, **kwargs):
    rows = make_trace(trace_id, **kwargs)
    attributes = json.loads(rows[0]["attributes"])
    attributes["gen_ai.input.messages"] = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": question or trace_id}]}]
    )
    attributes["gen_ai.output.messages"] = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": json.dumps({**ANSWER, "text": answer or trace_id})}
                ],
            }
        ]
    )
    rows[0]["attributes"] = json.dumps(attributes)
    attributes = json.loads(rows[1]["attributes"])
    attributes["gen_ai.tool.call.result"] = json.dumps(
        [{"type": "text", "text": evidence or f"Evidence for {trace_id}."}]
    )
    rows[1]["attributes"] = json.dumps(attributes)
    return rows


def checks(data_dir, items, name="initial"):
    directory = data_dir / "checks"
    directory.mkdir(exist_ok=True)
    rows = [
        {
            "trace_id": trace_id,
            "task_id": "task",
            "passed": passed,
            "criteria": json.dumps({"supported": passed}),
            "created_at": datetime.now(UTC),
        }
        for trace_id, passed in items
    ]
    pq.write_table(pa.Table.from_pylist(rows), directory / f"{name}.parquet")


def dataset(data_dir, items, overrides=None):
    data_dir.mkdir(exist_ok=True)
    (data_dir / "graders").mkdir(exist_ok=True)
    overrides = overrides or {}
    publish(
        data_dir,
        [
            row
            for index, (trace_id, _passed) in enumerate(items)
            for row in trace(trace_id, log_offset=index * 10, **overrides.get(trace_id, {}))
        ],
    )
    checks(data_dir, items)
    return Settings(data_dir)


def label_record(definition, *, text="Saved evidence", answer="yes", passed=False, trace_id="old"):
    state = {"rfi_question": text}
    return Label(
        grader=definition.name,
        grader_version=definition.version,
        question=definition.question,
        fields=definition.fields,
        applies_to=definition.applies_to,
        trace_id=trace_id,
        task_id="task",
        checker_passed=passed,
        labeled_at=datetime.now(UTC).isoformat(),
        state=state,
        input_hash=hash_state(state),
        label=answer,
    )


def write_record(directory, definition, document):
    directory.mkdir(exist_ok=True)
    path = directory / f"{definition.name}.jsonl"
    path.write_text(json.dumps(document) + "\n")
    return path


def test_shared_cohort_has_exact_projections_explicit_answers_and_odd_failure_quota(tmp_path):
    settings = dataset(tmp_path, [(f"trace-{i}", i >= 4) for i in range(8)])
    definitions = (
        grader("answered", ("rfi_question", "final_answer")),
        grader("grounded", ("final_answer", "retrieved_text")),
        grader("actionable", ("rfi_question",)),
    )
    labels_dir = tmp_path / "labels"
    shown, asked = [], []

    def ask(question):
        asked.append(question)
        return "no" if len(asked) % 2 else "yes"

    counts = label_traces(settings, definitions, 3, labels_dir, ask, shown.append)
    assert counts == {definition.name: 3 for definition in definitions}
    assert len(asked) == 9
    displayed_states = {}
    for display in shown:
        trace_line, _task_line, grader_line, state_json = display.split("\n", 3)
        displayed_states[(grader_line.split()[1], trace_line.removeprefix("Trace: "))] = json.loads(
            state_json
        )
    cohorts = []
    for definition in definitions:
        labels = read_labels(labels_dir, definition)
        cohorts.append({label.trace_id for label in labels})
        assert sum(not label.checker_passed for label in labels) == 2
        assert all(set(label.state) == set(definition.fields) for label in labels)
        assert all(label.input_hash == hash_state(label.state) for label in labels)
        assert all(label.task_id == "task" for label in labels)
        assert all(
            label.state == displayed_states[(definition.name, label.trace_id)] for label in labels
        )
    assert cohorts[0] == cohorts[1] == cohorts[2]
    assert len(cohorts[0]) == 3
    assert len({display.splitlines()[0] for display in shown}) == 3
    assert all("Checker" not in display for display in shown)
    assert {label.label for label in read_labels(labels_dir, definitions[0])} == {"yes", "no"}


def test_interrupt_saves_answer_and_resume_finishes_partial_trace_first(tmp_path):
    settings = dataset(tmp_path, [("failed", False), ("passed", True)])
    definitions = (grader(), grader("grounded", ("retrieved_text",)))
    labels_dir = tmp_path / "labels"
    attempts = []

    def interrupted(question):
        attempts.append(question)
        if len(attempts) == 2:
            raise EOFError
        return "yes"

    with pytest.raises(EOFError):
        label_traces(settings, definitions, 2, labels_dir, interrupted, lambda _: None)
    first = read_labels(labels_dir, definitions[0])
    assert len(first) == 1
    assert read_labels(labels_dir, definitions[1]) == []
    durable = (labels_dir / "answered.jsonl").read_bytes()
    shown, asked = [], []
    label_traces(
        settings,
        definitions,
        2,
        labels_dir,
        lambda question: asked.append(question) or "no",
        shown.append,
    )
    assert shown[0].startswith(f"Trace: {first[0].trace_id}\n")
    assert asked[0] == definitions[1].question
    assert len(asked) == 3
    assert (labels_dir / "answered.jsonl").read_bytes().startswith(durable)
    assert all(len(read_labels(labels_dir, definition)) == 2 for definition in definitions)
    assert {label.trace_id for label in read_labels(labels_dir, definitions[0])} == {
        label.trace_id for label in read_labels(labels_dir, definitions[1])
    }


def test_resume_uses_saved_trace_identity_when_another_trace_shares_one_projection(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("hopper.labels.random.shuffle", lambda _: None)
    settings = dataset(
        tmp_path,
        [("different", False), ("original", False), ("passed", True)],
        {
            "different": {"question": "a0", "evidence": "b2"},
            "original": {"question": "a0", "evidence": "b0"},
            "passed": {"question": "a1", "evidence": "b1"},
        },
    )
    definitions = (grader(), grader("grounded", ("retrieved_text",)))
    labels_dir = tmp_path / "labels"
    _append_label(
        labels_dir, definitions[0], label_record(definitions[0], text="a0", trace_id="original")
    )
    shown = []
    label_traces(settings, definitions, 2, labels_dir, lambda _: "yes", shown.append)
    assert shown[0].startswith("Trace: original\n")
    assert all(
        {label.trace_id for label in read_labels(labels_dir, definition)} == {"original", "passed"}
        for definition in definitions
    )


@pytest.mark.parametrize("original", [None, ("changed", False), ("a0", True)])
def test_resume_requires_the_original_saved_trace_snapshot_before_prompt(
    tmp_path, monkeypatch, original
):
    monkeypatch.setattr("hopper.labels.random.shuffle", lambda _: None)
    items = [("different", False), ("passed", True)]
    overrides = {
        "different": {"question": "a0", "evidence": "b2"},
        "passed": {"question": "a1", "evidence": "b1"},
    }
    if original is not None:
        question, passed = original
        items.append(("original", passed))
        overrides["original"] = {"question": question, "evidence": "b0"}
    settings = dataset(tmp_path, items, overrides)
    definitions = (grader(), grader("grounded", ("retrieved_text",)))
    labels_dir = tmp_path / "labels"
    _append_label(
        labels_dir, definitions[0], label_record(definitions[0], text="a0", trace_id="original")
    )
    original_bytes = (labels_dir / "answered.jsonl").read_bytes()
    shown = []
    with pytest.raises(ValueError, match="Cannot resume incomplete cohort trace original"):
        label_traces(
            settings,
            definitions,
            2,
            labels_dir,
            lambda question: shown.append(question) or "yes",
            shown.append,
        )
    assert shown == []
    assert (labels_dir / "answered.jsonl").read_bytes() == original_bytes
    assert not (labels_dir / "grounded.jsonl").exists()


def test_completed_target_does_not_open_corrupt_store_or_prompt(tmp_path):
    definition = grader()
    labels_dir = tmp_path / "labels"
    _append_label(labels_dir, definition, label_record(definition))
    spans = tmp_path / "spans"
    spans.mkdir()
    (spans / "broken.parquet").write_text("Not parquet")
    asked = []
    assert label_traces(
        Settings(tmp_path), [definition], 1, labels_dir, asked.append, asked.append
    ) == {"answered": 1}
    assert asked == []


def test_resume_deduplicates_projected_evidence_after_store_rebuild(tmp_path):
    definition = grader()
    labels_dir = tmp_path / "labels"
    first = dataset(tmp_path / "first", [("failed", False), ("passed", True)])
    label_traces(first, [definition], 2, labels_dir, lambda _: "yes", lambda _: None)
    original = read_labels(labels_dir, definition)
    rebuilt = dataset(
        tmp_path / "rebuilt",
        [("rebuilt-fail", False), ("rebuilt-pass", True), ("new-fail", False)],
        {"rebuilt-fail": {"question": "failed"}, "rebuilt-pass": {"question": "passed"}},
    )
    shown = []
    counts = label_traces(rebuilt, [definition], 3, labels_dir, lambda _: "no", shown.append)
    assert counts == {"answered": 3}
    assert len(shown) == 1
    assert shown[0].startswith("Trace: new-fail\n")
    assert read_labels(labels_dir, definition)[:2] == original


def test_preflight_counts_unique_evidence_and_prompts_nothing_on_shortage(tmp_path):
    settings = dataset(
        tmp_path,
        [("failed-one", False), ("failed-two", False), ("passed", True)],
        {"failed-one": {"question": "same"}, "failed-two": {"question": "same"}},
    )
    shown = []
    with pytest.raises(ValueError, match="insufficient unique projected states"):
        label_traces(settings, [grader()], 3, tmp_path / "labels", shown.append, shown.append)
    assert shown == []
    assert not (tmp_path / "labels").exists()


def test_overlapping_failure_pass_states_are_reserved_for_the_needed_pool(tmp_path):
    settings = dataset(
        tmp_path,
        [("failure-shared", False), ("failure-only", False), ("pass-shared", True)],
        {"failure-shared": {"question": "shared"}, "pass-shared": {"question": "shared"}},
    )
    definition = grader()
    label_traces(settings, [definition], 2, tmp_path / "labels", lambda _: "yes", lambda _: None)
    labels = read_labels(tmp_path / "labels", definition)
    assert {label.trace_id for label in labels} == {"failure-only", "pass-shared"}


def test_same_state_cannot_count_once_in_each_stratum(tmp_path):
    settings = dataset(
        tmp_path,
        [("failed", False), ("passed", True)],
        {"failed": {"question": "shared"}, "passed": {"question": "shared"}},
    )
    with pytest.raises(ValueError, match="disjoint"):
        label_traces(settings, [grader()], 2, tmp_path / "labels", lambda _: "yes", lambda _: None)
    assert not (tmp_path / "labels").exists()


def test_preflight_checks_all_graders_before_first_answer(tmp_path):
    settings = dataset(
        tmp_path,
        [("failed", False), ("passed", True)],
        {"failed": {"answer": "same"}, "passed": {"answer": "same"}},
    )
    asked = []
    with pytest.raises(ValueError, match="insufficient unique"):
        label_traces(
            settings,
            [grader(), grader("repeated", ("final_answer",))],
            2,
            tmp_path / "labels",
            asked.append,
            lambda _: None,
        )
    assert asked == []


def test_sampling_excludes_replays_evals_other_services_and_unfinished_traces(tmp_path):
    items = [
        ("failed", False),
        ("passed", True),
        ("replay", False),
        ("eval", False),
        ("other-service", False),
        ("unfinished", False),
        ("missing-check", False),
    ]
    settings = dataset(
        tmp_path,
        items,
        {
            "replay": {"replay_of": "failed"},
            "other-service": {"service": "other"},
            "unfinished": {"received_at": datetime.now(UTC)},
        },
    )
    rows = []
    for index, (trace_id, _passed) in enumerate(items):
        kwargs = {"log_offset": index * 10}
        if trace_id == "replay":
            kwargs["replay_of"] = "failed"
        if trace_id == "other-service":
            kwargs["service"] = "other"
        if trace_id == "unfinished":
            kwargs["received_at"] = datetime.now(UTC)
        current = trace(trace_id, **kwargs)
        if trace_id == "eval":
            for row in current:
                row["eval_run_id"] = "evaluation"
        rows.extend(current)
    publish(tmp_path, rows)
    checks(tmp_path, [item for item in items if item[0] != "missing-check"])
    definition = grader()
    label_traces(settings, [definition], 2, tmp_path / "labels", lambda _: "yes", lambda _: None)
    assert {label.trace_id for label in read_labels(tmp_path / "labels", definition)} == {
        "failed",
        "passed",
    }


def test_duplicate_checker_rows_fail_before_prompt(tmp_path):
    settings = dataset(tmp_path, [("failed", False), ("passed", True)])
    checks(tmp_path, [("failed", False)], name="duplicate")
    asked = []
    with pytest.raises(ValueError, match="Duplicate stored check"):
        label_traces(settings, [grader()], 2, tmp_path / "labels", asked.append, lambda _: None)
    assert asked == []


def test_joint_cohort_requires_shared_applicability(tmp_path):
    settings = dataset(tmp_path, [("failed", False), ("passed", True)])
    definitions = [grader(), replace(grader("other"), applies_to="other-service")]
    with pytest.raises(ValueError, match="no shared applicable service"):
        label_traces(settings, definitions, 2, tmp_path / "labels", lambda _: "yes", lambda _: None)


def test_fifty_trace_cohort_is_shared_and_unique_for_all_three_graders(tmp_path):
    settings = dataset(tmp_path, [(f"trace-{i}", i >= 50) for i in range(100)])
    definitions = [
        grader(),
        grader("grounded", ("final_answer", "retrieved_text")),
        grader("actionable", ("rfi_question", "final_answer")),
    ]
    asked = []
    counts = label_traces(
        settings,
        definitions,
        50,
        tmp_path / "labels",
        lambda question: asked.append(question) or "yes",
        lambda _: None,
    )
    assert counts == {definition.name: 50 for definition in definitions}
    assert len(asked) == 150
    cohorts = []
    for definition in definitions:
        labels = read_labels(tmp_path / "labels", definition)
        assert len({label.input_hash for label in labels}) == 50
        assert sum(not label.checker_passed for label in labels) == 25
        cohorts.append({label.trace_id for label in labels})
    assert len(cohorts[0]) == 50
    assert cohorts[0] == cohorts[1] == cohorts[2]


def test_joint_cohort_backtracks_when_first_failure_blocks_every_shared_pass(tmp_path, monkeypatch):
    monkeypatch.setattr("hopper.labels.random.shuffle", lambda _: None)
    settings = dataset(
        tmp_path,
        [("f0", False), ("f1", False), ("p1", True), ("p2", True)],
        {
            "f0": {"question": "a0", "evidence": "b0"},
            "f1": {"question": "a2", "evidence": "b2"},
            "p1": {"question": "a1", "evidence": "b0"},
            "p2": {"question": "a0", "evidence": "b1"},
        },
    )
    definitions = (grader(), grader("grounded", ("retrieved_text",)))
    labels_dir = tmp_path / "labels"
    shown = []
    counts = label_traces(settings, definitions, 2, labels_dir, lambda _: "yes", shown.append)
    assert counts == {definition.name: 2 for definition in definitions}
    assert len(shown) == 4
    for definition in definitions:
        labels = read_labels(labels_dir, definition)
        assert {label.trace_id for label in labels} == {"f1", "p1"}
        assert len({label.input_hash for label in labels}) == 2
        assert sum(not label.checker_passed for label in labels) == 1


def test_cohort_search_limit_reports_unknown_feasibility_before_any_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("hopper.labels.random.shuffle", lambda _: None)
    monkeypatch.setattr("hopper.labels._MAX_PLAN_SEARCH_STEPS", 1)
    settings = dataset(tmp_path, [("failed", False), ("passed", True)])
    shown = []
    with pytest.raises(ValueError, match="search limit.*feasibility is unknown"):
        label_traces(settings, [grader()], 2, tmp_path / "labels", shown.append, shown.append)
    assert shown == []
    assert not (tmp_path / "labels").exists()


def test_jointly_unfillable_cohort_fails_before_any_prompt(tmp_path):
    settings = dataset(
        tmp_path,
        [("f-one", False), ("f-two", False), ("p-one", True), ("p-two", True)],
        {
            "f-one": {"question": "q-one", "evidence": "text-one"},
            "f-two": {"question": "q-two", "evidence": "text-two"},
            "p-one": {"question": "q-one", "evidence": "text-two"},
            "p-two": {"question": "q-two", "evidence": "text-one"},
        },
    )
    asked = []
    with pytest.raises(ValueError, match="Could not form a shared"):
        label_traces(
            settings,
            [grader(), grader("grounded", ("retrieved_text",))],
            2,
            tmp_path / "labels",
            asked.append,
            lambda _: None,
        )
    assert asked == []
    assert not (tmp_path / "labels").exists()


def test_no_implicit_answer_is_saved_for_invalid_callback_value(tmp_path):
    settings = dataset(tmp_path, [("failed", False)])
    with pytest.raises(ValueError, match="explicit yes or no"):
        label_traces(settings, [grader()], 1, tmp_path / "labels", lambda _: "y", lambda _: None)
    assert not (tmp_path / "labels").exists()


def test_locked_duplicate_append_is_idempotent_across_concurrent_writers(tmp_path):
    definition = grader()
    record = label_record(definition)
    with ThreadPoolExecutor(max_workers=8) as executor:
        appended = list(
            executor.map(lambda _: _append_label(tmp_path, definition, record), range(16))
        )
    assert sum(appended) == 1
    assert read_labels(tmp_path, definition) == [record]
    assert (tmp_path / "answered.jsonl").read_bytes().count(b"\n") == 1


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_unicode_line_separators_round_trip_in_label_evidence(tmp_path, separator):
    definition = grader()
    record = label_record(definition, text=f"Before{separator}after")
    assert _append_label(tmp_path, definition, record)
    assert read_labels(tmp_path, definition) == [record]


def test_conflicting_append_fails_and_preserves_original_bytes(tmp_path):
    definition = grader()
    record = label_record(definition)
    _append_label(tmp_path, definition, record)
    path = tmp_path / "answered.jsonl"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        _append_label(tmp_path, definition, replace(record, label="no", trace_id="rebuilt"))
    assert path.read_bytes() == original


def test_reader_deduplicates_equivalent_evidence_across_versions_and_trace_ids(tmp_path):
    definition = grader()
    record = label_record(definition)
    path = write_record(tmp_path, definition, asdict(record))
    with path.open("a") as file:
        file.write(json.dumps(asdict(replace(record, grader_version=2, trace_id="rebuilt"))) + "\n")
    changed = replace(definition, version=3, model="jev-2.0.0", yes_threshold=0.8, no_threshold=0.2)
    assert read_labels(tmp_path, changed) == [record]


def test_reader_rejects_conflicting_duplicate_records(tmp_path):
    definition = grader()
    record = label_record(definition)
    path = write_record(tmp_path, definition, asdict(record))
    with path.open("a") as file:
        file.write(json.dumps(asdict(replace(record, label="no"))) + "\n")
    with pytest.raises(ValueError, match="line 2.*Conflicting duplicate"):
        read_labels(tmp_path, definition)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"checker_passed": 0}, "checker_passed"),
        ({"grader_version": True}, "grader_version"),
        ({"label": "unsure"}, "label must be yes or no"),
        ({"input_hash": "a" * 64}, "input_hash"),
        ({"state": {"rfi_question": "Saved evidence", "needs_engineer": None}}, "exactly"),
        ({"labeled_at": "2026-11-11"}, "timezone"),
        ({"task_id": 2}, "task_id"),
        ({"unexpected": True}, "unknown fields"),
        ({"question": "Different criterion"}, "Incompatible"),
        ({"applies_to": "other-service"}, "Incompatible"),
    ],
)
def test_reader_rejects_malformed_or_incompatible_evidence(tmp_path, change, match):
    definition = grader()
    path = write_record(tmp_path, definition, {**asdict(label_record(definition)), **change})
    with pytest.raises(ValueError, match=match) as error:
        read_labels(tmp_path, definition)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("fragment", ['{"grader":', '{"valid": true}', "\n"])
def test_interrupted_or_invalid_tail_fails_without_truncating_user_data(tmp_path, fragment):
    definition = grader()
    record = label_record(definition)
    path = write_record(tmp_path, definition, asdict(record))
    with path.open("a") as file:
        file.write(fragment)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid label file"):
        read_labels(tmp_path, definition)
    with pytest.raises(ValueError, match="Invalid label file"):
        _append_label(tmp_path, definition, replace(record, trace_id="new"))
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "contents,match",
    [
        ('{"grader":"answered","grader":"answered"}\n', "Duplicate JSON key"),
        ('{"state":{"rfi_question":NaN}}\n', "Non-finite"),
        ('{"state":{"rfi_question":1e999}}\n', "Non-finite"),
    ],
)
def test_reader_rejects_duplicate_json_keys_and_nonfinite_values(tmp_path, contents, match):
    (tmp_path / "answered.jsonl").write_text(contents)
    with pytest.raises(ValueError, match=match):
        read_labels(tmp_path, grader())


def test_cli_three_questions_invalid_reply_quit_and_resume(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    settings = dataset(data_dir, [("failed", False), ("passed", True)])
    Catalog(data_dir).install_seeds(lambda: MODEL)
    monkeypatch.setenv("HOPPER_DATA_DIR", str(data_dir))
    labels_dir = tmp_path / "labels"
    names = ["answered_question", "grounded", "actionable"]
    command = ["label", "--n", "2", "--labels-dir", str(labels_dir)]
    for name in names:
        command.extend(["--grader", name])
    runner = CliRunner()
    first = runner.invoke(app, command, input="maybe\nYeS\nquit\n")
    assert first.exit_code == 0, first.output
    assert "Enter yes, no, or quit." in first.output
    assert "Completed answers were saved" in first.output
    definitions = {item.name: item for item in Catalog(settings.data_dir).list_graders()}
    saved = read_labels(labels_dir, definitions[names[0]])
    assert len(saved) == 1
    assert all(read_labels(labels_dir, definitions[name]) == [] for name in names[1:])
    assert set(saved[0].state) == {"rfi_question", "final_answer"}
    assert "retrieved_text" not in saved[0].state
    second = runner.invoke(app, command, input="n\ny\nYES\nNO\ny\n")
    assert second.exit_code == 0, second.output
    assert "grader\tlabels" in second.output
    assert all(len(read_labels(labels_dir, definitions[name])) == 2 for name in names)
    assert (
        len(
            {
                label.trace_id
                for name in names
                for label in read_labels(labels_dir, definitions[name])
            }
        )
        == 2
    )
    assert read_labels(labels_dir, definitions[names[0]])[0] == saved[0]
    assert "Checker passed" not in first.output + second.output


@pytest.mark.parametrize(
    "names,match",
    [
        (["unknown"], "No non-retired grader"),
        (["cites_sources"], "code grader"),
        (["grounded", "grounded"], "only once"),
    ],
)
def test_cli_rejects_invalid_graders_before_prompt(tmp_path, monkeypatch, names, match):
    (tmp_path / "graders").mkdir()
    Catalog(tmp_path).install_seeds(lambda: MODEL)
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    command = ["label", "--n", "1", "--labels-dir", str(tmp_path / "labels")]
    for name in names:
        command.extend(["--grader", name])
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 1
    assert match in result.output
    assert "[yes/no/quit]" not in result.output
    assert not (tmp_path / "labels").exists()
