import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from hopper.corpus import load_corpus
from hopper.graders.state import MAX_RETRIEVED_CHARACTERS, rfi_core

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
ANSWER = {"text": "Use the specification.", "citations": ["SPEC-01/1.1"], "needs_engineer": False}
INPUT = "gen_ai.input.messages"
OUTPUT = "gen_ai.output.messages"
ARGUMENTS = "gen_ai.tool.call.arguments"
RESULT = "gen_ai.tool.call.result"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def make_message(role, text):
    return {"role": role, "parts": [{"type": "text", "content": text}]}


def make_span(
    span_id, *, parent="root", tool=None, arguments=None, result=None, attributes=None, position=0
):
    values = dict(attributes or {})
    if arguments is not None:
        values[ARGUMENTS] = arguments
    if result is not None:
        values[RESULT] = result
    return {
        "trace_id": "trace",
        "span_id": span_id,
        "parent_span_id": parent,
        "start_time": EPOCH + timedelta(seconds=position),
        "operation": "execute_tool" if tool else "invoke_agent",
        "tool_name": tool,
        "status": "OK",
        "error_type": None,
        "received_at": EPOCH,
        "log_offset": 0,
        "attributes": json.dumps({key: json.dumps(value) for key, value in values.items()}),
    }


def test_latest_question_and_full_submission_are_authoritative(corpus):
    latest = {
        "role": "user",
        "parts": [
            {"type": "text", "content": "Follow-up: "},
            {"type": "uri", "uri": "corpus://drawings/M-101.png"},
            {"type": "text", "content": "what changed?"},
        ],
    }
    root = make_span(
        "root",
        parent=None,
        attributes={
            INPUT: [
                make_message("user", "Original question"),
                make_message("assistant", "Earlier answer"),
                latest,
            ],
            OUTPUT: [make_message("assistant", json.dumps({"text": "Different root answer"}))],
        },
    )
    answer = {**ANSWER, "value": 42}
    submission = make_span("submit", tool="submit_answer", arguments=answer, position=1)
    state = rfi_core([submission, root], corpus)
    assert state == {
        "rfi_question": "Follow-up: what changed?",
        "final_answer": answer,
        "retrieved_text": "",
        "tool_calls": [{"name": "submit_answer", "arguments": answer}],
        "needs_engineer": False,
    }


def test_root_answer_fallback_does_not_use_research_child(corpus):
    root = make_span(
        "root",
        parent=None,
        attributes={OUTPUT: [make_message("assistant", json.dumps(ANSWER))]},
    )
    child = make_span(
        "research",
        attributes={OUTPUT: [make_message("assistant", '{"findings":"Not the final answer"}')]},
    )
    assert rfi_core([root, child], corpus)["final_answer"] == ANSWER
    orphan = rfi_core([child], corpus)
    assert orphan["final_answer"] is None
    assert orphan["rfi_question"] is None
    assert orphan["needs_engineer"] is None


@pytest.mark.parametrize(("status", "error_type"), [("ERROR", None), ("UNSET", "TimeoutError")])
def test_failed_tools_remain_attempts_without_evidence_or_answers(corpus, status, error_type):
    failed_read = make_span(
        "read", tool="read_section", arguments={"id": "SPEC-01"}, result="malformed result"
    )
    failed_submission = make_span("submit", tool="submit_answer", arguments=ANSWER)
    for span in (failed_read, failed_submission):
        span.update(status=status, error_type=error_type)
    state = rfi_core([failed_read, failed_submission], corpus)
    assert state["retrieved_text"] == ""
    assert state["final_answer"] is None
    assert [call["name"] for call in state["tool_calls"]] == ["read_section", "submit_answer"]


def test_successful_retry_supplies_answer_and_keeps_failed_attempt(corpus):
    failed = make_span("first", tool="submit_answer", arguments={"text": "Rejected"})
    failed["status"] = "ERROR"
    successful = make_span("second", tool="submit_answer", arguments=ANSWER, position=1)
    state = rfi_core([successful, failed], corpus)
    assert state["final_answer"] == ANSWER
    assert len(state["tool_calls"]) == 2


def test_retrieval_order_drawings_repeated_reads_and_duplicate_uploads(corpus):
    search_text = json.dumps([{"id": "SPEC-01/1.1", "text": "Search evidence"}])
    rows = [
        make_span(
            "a",
            tool="search_documents",
            arguments={"query": "work"},
            result=[{"type": "text", "text": search_text}],
        ),
        make_span(
            "b",
            tool="read_section",
            arguments={"id": "SPEC-01/1.1"},
            result=[{"type": "text", "text": "Section evidence"}],
        ),
        make_span(
            "c",
            tool="list_addenda",
            arguments={},
            result=[{"type": "text", "text": "Addenda evidence"}],
            position=1,
        ),
        make_span(
            "d",
            tool="read_drawing",
            arguments={"id": "M-101"},
            result=[
                {"type": "text", "text": "Drawing M-101"},
                {
                    "type": "uri",
                    "modality": "image",
                    "mime_type": "image/png",
                    "uri": "corpus://drawings/M-101.png",
                },
            ],
            position=2,
        ),
        make_span(
            "e",
            tool="read_section",
            arguments={"id": "SPEC-01/1.1"},
            result=[{"type": "text", "text": "Section evidence"}],
            position=3,
        ),
    ]
    state = rfi_core(rows, corpus)
    assert state["retrieved_text"] == "\n\n".join(
        [
            search_text,
            "Section evidence",
            "Addenda evidence",
            "Drawing M-101",
            corpus.documents["M-101"],
            "Section evidence",
        ]
    )
    assert [call["name"] for call in state["tool_calls"]] == [
        "search_documents",
        "read_section",
        "list_addenda",
        "read_drawing",
        "read_section",
    ]
    duplicate = copy.deepcopy(rows[0])
    duplicate.update(received_at=EPOCH + timedelta(hours=1), log_offset=999)
    attributes = json.loads(duplicate["attributes"])
    attributes[ARGUMENTS] = json.dumps(json.loads(attributes[ARGUMENTS]), indent=2)
    duplicate["attributes"] = json.dumps(attributes, indent=2, sort_keys=True)
    assert rfi_core([duplicate, *reversed(rows)], corpus) == state


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        (ARGUMENTS, {"id": "different"}),
        (RESULT, [{"type": "text", "text": "Different evidence"}]),
    ],
)
def test_conflicting_duplicate_spans_fail(corpus, attribute, value):
    original = make_span(
        "duplicate",
        tool="read_section",
        arguments={"id": "SPEC-01"},
        result=[{"type": "text", "text": "Evidence"}],
    )
    changed = copy.deepcopy(original)
    attributes = json.loads(changed["attributes"])
    attributes[attribute] = json.dumps(value)
    changed["attributes"] = json.dumps(attributes)
    with pytest.raises(ValueError, match="Trace trace, span duplicate, duplicate"):
        rfi_core([original, changed], corpus)


def test_retrieved_text_is_an_ordered_prefix(corpus):
    text = "A" * (MAX_RETRIEVED_CHARACTERS - 3) + "abcdef"
    row = make_span(
        "read",
        tool="read_section",
        arguments={"id": "SPEC-01"},
        result=[{"type": "text", "text": text}],
    )
    state = rfi_core([row], corpus)
    assert state["retrieved_text"] == text[:MAX_RETRIEVED_CHARACTERS]
    assert state["tool_calls"] == [{"name": "read_section", "arguments": {"id": "SPEC-01"}}]


@pytest.mark.parametrize(
    ("attribute", "raw"),
    [
        ("attributes", "["),
        ("attributes", "[]"),
        (ARGUMENTS, "["),
        (ARGUMENTS, "[]"),
        (ARGUMENTS, '{"value":NaN}'),
        (ARGUMENTS, '{"value":1e400}'),
        (RESULT, "{}"),
        (RESULT, '[{"type":"text","text":7}]'),
    ],
)
def test_malformed_consumed_data_names_trace_span_and_attribute(corpus, attribute, raw):
    row = make_span("bad", tool="read_section", arguments={"id": "SPEC-01"})
    if attribute == "attributes":
        row["attributes"] = raw
    else:
        attributes = json.loads(row["attributes"])
        attributes[attribute] = raw
        row["attributes"] = json.dumps(attributes)
    with pytest.raises(ValueError, match=f"Trace trace, span bad, {attribute}"):
        rfi_core([row], corpus)


@pytest.mark.parametrize("uri", ["https://example.com/M-101.png", "corpus://drawings/X-999.png"])
def test_drawing_references_must_belong_to_the_corpus(corpus, uri):
    row = make_span("drawing", tool="read_drawing", result=[{"type": "uri", "uri": uri}])
    with pytest.raises(ValueError, match="unknown drawing reference"):
        rfi_core([row], corpus)


def test_malformed_submission_does_not_fall_back_to_root_answer(corpus):
    root = make_span(
        "root", parent=None, attributes={OUTPUT: [make_message("assistant", json.dumps(ANSWER))]}
    )
    malformed = make_span("submit", tool="submit_answer", arguments=[])
    with pytest.raises(ValueError, match="span submit, gen_ai.tool.call.arguments"):
        rfi_core([root, malformed], corpus)


@pytest.mark.parametrize("kind", ["roots", "submissions", "traces"])
def test_ambiguous_trace_identity_fails(corpus, kind):
    if kind == "roots":
        rows = [make_span("a", parent=None), make_span("b", parent=None)]
    elif kind == "submissions":
        rows = [
            make_span("a", tool="submit_answer", arguments=ANSWER),
            make_span("b", tool="submit_answer", arguments=ANSWER),
        ]
    else:
        rows = [make_span("a"), make_span("b")]
        rows[1]["trace_id"] = "another-trace"
    with pytest.raises(ValueError, match="multiple|exactly one trace"):
        rfi_core(rows, corpus)


@pytest.mark.parametrize("arguments", [[], None])
def test_invalid_failed_call_arguments_do_not_hide_a_successful_answer(corpus, arguments):
    failed = make_span(
        "failed",
        tool="read_section",
        attributes={ARGUMENTS: arguments},
        result=[{"type": "text", "text": "ValueError: arguments must be an object"}],
    )
    failed["status"] = "ERROR"
    successful = make_span("submit", tool="submit_answer", arguments=ANSWER, position=1)
    state = rfi_core([failed, successful], corpus)
    assert state["final_answer"] == ANSWER
    assert state["retrieved_text"] == ""
    assert state["tool_calls"][0] == {"name": "read_section", "arguments": arguments}
