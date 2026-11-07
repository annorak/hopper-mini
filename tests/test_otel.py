import base64
import gzip
import json

import pytest
import requests
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

from hopper.config import Settings
from hopper.otel import (
    Image,
    Telemetry,
    json_content,
    record_error,
    text_message,
    trace_content,
    trace_messages,
    wire_content,
)


@pytest.fixture
def requests_sent(monkeypatch):
    sent = []

    def send(session, request, **kwargs):
        sent.append(request)
        response = requests.Response()
        response.status_code = 200
        response._content = b""
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return sent


def decode(request):
    assert request.url == "http://collector.invalid/v1/traces"
    assert request.headers["Content-Encoding"] == "gzip"
    assert request.headers["Content-Type"] == "application/x-protobuf"
    payload = gzip.decompress(request.body)
    return ExportTraceServiceRequest.FromString(payload)


def attributes(values):
    result = {}
    for item in values:
        value = item.value
        kind = value.WhichOneof("value")
        if kind == "array_value":
            result[item.key] = [
                getattr(entry, entry.WhichOneof("value")) for entry in value.array_value.values
            ]
        else:
            result[item.key] = getattr(value, kind)
    return result


def settings(tmp_path):
    return Settings(tmp_path, endpoint="http://collector.invalid/")


def test_images_reach_provider_and_only_references_reach_traces(requests_sent):
    png = b"\x89PNG\r\n\x1a\nprivate-drawing-bytes"
    image = Image("M-101", png)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "drawing-1",
                    "content": [{"type": "text", "text": "Sheet M-101"}, image],
                }
            ],
        }
    ]
    requests.post("https://anthropic.invalid/v1/messages", json=wire_content(messages))
    received = json.loads(requests_sent[0].body)
    source = received[0]["content"][0]["content"][1]["source"]
    assert source["media_type"] == "image/png"
    assert base64.b64decode(source["data"]) == png
    expected = {
        "type": "uri",
        "modality": "image",
        "mime_type": "image/png",
        "uri": "corpus://drawings/M-101.png",
    }
    assert trace_content(messages)[0]["content"][0]["content"][1] == expected
    assert trace_messages(messages)[0]["parts"][0] == {
        "type": "tool_call_response",
        "id": "drawing-1",
        "response": [{"type": "text", "text": "Sheet M-101"}, expected],
    }
    assert source["data"] not in json_content(messages)
    assert messages[0]["content"][0]["content"][1] is image
    assert wire_content({"deeper": ([image],)})["deeper"][0][0]["source"] == source


@pytest.mark.parametrize("kind", ["native", "blob"])
def test_raw_base64_image_capture_is_rejected(kind):
    block = (
        {"type": "image", "source": {"type": "base64", "data": "secret"}}
        if kind == "native"
        else {"type": "blob", "modality": "image", "content": "secret"}
    )
    with pytest.raises(ValueError, match="Trace images must use Image"):
        json_content({"nested": [{"type": "tool_result", "content": [block]}]})
    with pytest.raises(ValueError, match="Trace images must use Image"):
        trace_messages([{"role": "user", "content": [block]}])


def test_messages_convert_text_thinking_and_tool_requests():
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Check the addendum first", "signature": "opaque"},
            {"type": "redacted_thinking", "data": "opaque-redacted"},
            {"type": "text", "text": "Reading the section."},
            {"type": "tool_use", "id": "read-1", "name": "read_section", "input": {"id": "23"}},
        ],
    }
    parts = trace_messages([message])[0]["parts"]
    assert parts == [
        {"type": "reasoning", "content": "Check the addendum first"},
        {"type": "generic", "value": {"type": "redacted_thinking"}},
        {"type": "text", "content": "Reading the section."},
        {"type": "tool_call", "id": "read-1", "name": "read_section", "arguments": {"id": "23"}},
    ]
    assert "opaque" not in json_content(parts)
    assert wire_content([message]) == [message]
    assert trace_messages([text_message("user", "Question")]) == [
        {"role": "user", "parts": [{"type": "text", "content": "Question"}]}
    ]
    assert trace_messages([{"role": "user", "content": "Question"}]) == [
        {"role": "user", "parts": [{"type": "text", "content": "Question"}]}
    ]


def test_actual_otlp_preserves_nested_spans_resources_and_json_content(tmp_path, requests_sent):
    attrs = {
        "gen_ai.input.messages": json_content(
            trace_messages([text_message("user", "RFI question")])
        ),
        "gen_ai.output.messages": json_content(
            trace_messages(
                [{"role": "assistant", "content": [{"type": "thinking", "thinking": "Summary"}]}]
            )
        ),
        "gen_ai.system_instructions": json_content([{"type": "text", "text": "Instructions"}]),
        "gen_ai.tool.definitions": json_content([{"name": "read_drawing", "input_schema": {}}]),
        "gen_ai.tool.call.arguments": json_content({"drawing_id": "M-101"}),
        "gen_ai.tool.call.result": json_content({"nested": [Image("M-101", b"PRIVATE_PNG")]}),
        "gen_ai.response.model": "claude-haiku-5-5",
        "gen_ai.usage.input_tokens": 11,
        "gen_ai.usage.output_tokens": 12,
        "gen_ai.usage.cache_read.input_tokens": 13,
        "gen_ai.usage.cache_creation.input_tokens": 14,
        "gen_ai.response.finish_reasons": ["tool_use"],
        "gen_ai.tool.call.id": "drawing-1",
    }
    with Telemetry(settings(tmp_path), "RFI-001", "baseline") as telemetry:
        with telemetry.span("invoke_agent", "rfi", conversation_id="conversation-1", root=True):
            with telemetry.span("invoke_agent", "research", conversation_id="conversation-1"):
                with telemetry.span(
                    "chat", "claude-haiku-5-5", conversation_id="conversation-1", attributes=attrs
                ):
                    pass
                with telemetry.span(
                    "execute_tool", "read_drawing", conversation_id="conversation-1"
                ) as tool:
                    record_error(tool, ValueError("seeded tool fault"))
    assert len(requests_sent) == 1
    exported = decode(requests_sent[0])
    assert len(exported.resource_spans) == 1
    resource = exported.resource_spans[0]
    assert attributes(resource.resource.attributes) == {
        "service.name": "rfi-agent",
        "hopper.task.id": "RFI-001",
        "hopper.agent.version": "baseline",
    }
    spans = {span.name: span for scope in resource.scope_spans for span in scope.spans}
    root = spans["invoke_agent rfi"]
    research = spans["invoke_agent research"]
    chat = spans["chat claude-haiku-5-5"]
    tool = spans["execute_tool read_drawing"]
    assert root.parent_span_id == b""
    assert research.parent_span_id == root.span_id
    assert chat.parent_span_id == research.span_id
    assert tool.parent_span_id == research.span_id
    assert len({span.trace_id for span in spans.values()}) == 1
    assert root.kind == research.kind == tool.kind == Span.SPAN_KIND_INTERNAL
    assert chat.kind == Span.SPAN_KIND_CLIENT
    for span in spans.values():
        values = attributes(span.attributes)
        operation, identity = span.name.split(" ", 1)
        assert values["gen_ai.operation.name"] == operation
        assert values["gen_ai.conversation.id"] == "conversation-1"
        assert values["gen_ai.provider.name"] == "anthropic"
    chat_attrs = attributes(chat.attributes)
    assert chat_attrs.items() >= attrs.items()
    for key in attrs:
        if key.endswith(("messages", "instructions", "definitions", "arguments", "result")):
            assert isinstance(chat_attrs[key], str)
            json.loads(chat_attrs[key])
    assert json.loads(chat_attrs["gen_ai.output.messages"])[0]["parts"] == [
        {"type": "reasoning", "content": "Summary"}
    ]
    assert "PRIVATE_PNG" not in str(exported)
    assert "corpus://drawings/M-101.png" in chat_attrs["gen_ai.tool.call.result"]
    assert tool.status.code == Status.STATUS_CODE_ERROR
    assert attributes(tool.attributes)["error.type"] == "ValueError"
    assert [event.name for event in tool.events] == ["exception"]
    assert attributes(tool.events[0].attributes)["exception.message"] == "seeded tool fault"


def test_task_resources_and_roots_are_isolated(tmp_path, requests_sent):
    with Telemetry(settings(tmp_path), "RFI-A", "v1") as first:
        with first.span("invoke_agent", "rfi", conversation_id="same", root=True) as outer:
            with Telemetry(
                settings(tmp_path),
                "RFI-B",
                "v2",
                resource_attributes={"hopper.eval.run_id": "eval-1"},
            ) as second:
                with second.span("invoke_agent", "rfi", conversation_id="same", root=True) as inner:
                    assert inner.get_span_context().trace_id != outer.get_span_context().trace_id
            with first.span("chat", "model", conversation_id="same"):
                pass
    assert len(requests_sent) == 2
    resources = {}
    for request in requests_sent:
        resource = decode(request).resource_spans[0]
        values = attributes(resource.resource.attributes)
        resources[values["hopper.task.id"]] = (
            values,
            [span for scope in resource.scope_spans for span in scope.spans],
        )
    assert resources["RFI-A"][0]["hopper.agent.version"] == "v1"
    assert resources["RFI-B"][0]["hopper.agent.version"] == "v2"
    assert resources["RFI-B"][0]["hopper.eval.run_id"] == "eval-1"
    assert "hopper.eval.run_id" not in resources["RFI-A"][0]
    for _, spans in resources.values():
        root = next(span for span in spans if span.name == "invoke_agent rfi")
        assert not root.parent_span_id
        assert all(span.trace_id == root.trace_id for span in spans)


@pytest.mark.parametrize("identity", ["service.name", "hopper.task.id", "hopper.agent.version"])
def test_resource_identity_cannot_be_overridden(tmp_path, identity):
    with pytest.raises(ValueError, match="identity cannot be replaced"):
        Telemetry(settings(tmp_path), "RFI-A", "v1", resource_attributes={identity: "replacement"})


def test_escaping_exception_marks_failed_spans_once(tmp_path, requests_sent):
    with pytest.raises(LookupError, match="no drawing"):
        with Telemetry(settings(tmp_path), "RFI-001", "baseline") as telemetry:
            with telemetry.span("invoke_agent", "rfi", conversation_id="one", root=True):
                with telemetry.span("execute_tool", "read_drawing", conversation_id="one"):
                    raise LookupError("no drawing")
    exported = decode(requests_sent[0])
    spans = exported.resource_spans[0].scope_spans[0].spans
    assert len(spans) == 2
    for span in spans:
        assert span.status.code == Status.STATUS_CODE_ERROR
        assert attributes(span.attributes)["error.type"] == "LookupError"
        assert [event.name for event in span.events] == ["exception"]


@pytest.mark.parametrize("is_exception", [False, True])
def test_export_failure_is_visible_on_close(tmp_path, monkeypatch, is_exception):
    def send(session, request, **kwargs):
        if is_exception:
            raise ValueError("HTTP transport failed")
        response = requests.Response()
        response.status_code = 400
        response._content = b"rejected"
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(RuntimeError, match="telemetry could not be exported") as caught:
        with Telemetry(settings(tmp_path), "RFI-001", "baseline") as telemetry:
            with telemetry.span("invoke_agent", "rfi", conversation_id="one", root=True):
                pass
    assert caught.value.__cause__ is not None


def test_export_failure_never_masks_original_error_and_closes_session(tmp_path, monkeypatch):
    closed = []
    monkeypatch.setattr(requests.Session, "close", lambda session: closed.append(session))

    def send(session, request, **kwargs):
        raise ValueError("HTTP transport failed")

    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(LookupError, match="original task failure") as caught:
        with Telemetry(settings(tmp_path), "RFI-001", "baseline") as telemetry:
            with telemetry.span("invoke_agent", "rfi", conversation_id="one", root=True):
                raise LookupError("original task failure")
    assert caught.value.__notes__ == [
        "Telemetry cleanup failed: Task telemetry could not be exported"
    ]
    assert len(closed) == 1
    telemetry.close()
    assert len(closed) == 1
    with pytest.raises(RuntimeError, match="telemetry is closed"):
        with telemetry.span("invoke_agent", "rfi", conversation_id="one", root=True):
            pass


@pytest.mark.parametrize("status", [200, 302])
def test_collector_rejection_cannot_report_success(tmp_path, monkeypatch, status):
    reply = ExportTraceServiceResponse()
    reply.partial_success.rejected_spans = 1
    reply.partial_success.error_message = "rejected"

    def send(session, request, **kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = reply.SerializeToString()
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(RuntimeError, match="telemetry could not be exported"):
        with Telemetry(settings(tmp_path), "RFI-001", "baseline") as telemetry:
            with telemetry.span("invoke_agent", "rfi", conversation_id="one", root=True):
                pass
