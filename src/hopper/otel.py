import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import requests
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from hopper.config import Settings


@dataclass(frozen=True)
class Image:
    drawing_id: str
    png: bytes


def wire_content(value: Any) -> Any:
    if isinstance(value, Image):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(value.png).decode("ascii"),
            },
        }
    if isinstance(value, dict):
        return {key: wire_content(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire_content(item) for item in value]
    return value


def trace_content(value: Any) -> Any:
    if isinstance(value, Image):
        return {
            "type": "uri",
            "modality": "image",
            "mime_type": "image/png",
            "uri": f"corpus://drawings/{value.drawing_id}.png",
        }
    if isinstance(value, dict):
        if value.get("type") == "image" or (
            value.get("type") == "blob" and value.get("modality") == "image"
        ):
            raise ValueError("Trace images must use Image with a corpus drawing ID")
        return {key: trace_content(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [trace_content(item) for item in value]
    return value


def json_content(value: Any) -> str:
    return json.dumps(trace_content(value), ensure_ascii=False, separators=(",", ":"))


def text_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _trace_part(block: Any) -> dict:
    if isinstance(block, Image):
        return trace_content(block)
    match block["type"]:
        case "text":
            return {"type": "text", "content": block["text"]}
        case "thinking":
            return {"type": "reasoning", "content": block["thinking"]}
        case "tool_use":
            return {
                "type": "tool_call",
                "id": block["id"],
                "name": block["name"],
                "arguments": trace_content(block["input"]),
            }
        case "tool_result":
            return {
                "type": "tool_call_response",
                "id": block["tool_use_id"],
                "response": trace_content(block["content"]),
            }
        case "redacted_thinking":
            return {"type": "generic", "value": {"type": "redacted_thinking"}}
        case _:
            return {"type": "generic", "value": trace_content(block)}


def trace_messages(messages: Sequence[dict]) -> list[dict]:
    return [
        {
            "role": message["role"],
            "parts": (
                [{"type": "text", "content": message["content"]}]
                if isinstance(message["content"], str)
                else [_trace_part(block) for block in message["content"]]
            ),
        }
        for message in messages
    ]


def record_error(span: Span, error: BaseException) -> None:
    span.set_attribute("error.type", type(error).__name__)
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.record_exception(error)


class _CheckedExporter(SpanExporter):
    def __init__(self, exporter: SpanExporter):
        self.exporter = exporter
        self.error: Exception | None = None

    def export(self, spans) -> SpanExportResult:
        try:
            result = self.exporter.export(spans)
        except Exception as error:
            self.error = error
            return SpanExportResult.FAILURE
        if result is not SpanExportResult.SUCCESS:
            self.error = RuntimeError("OTLP trace export failed")
        return result

    def shutdown(self) -> None:
        self.exporter.shutdown()


class _TraceSession(requests.Session):
    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        if 200 <= response.status_code < 400 and response.status_code != 200:
            raise RuntimeError(f"OTLP returned unexpected HTTP {response.status_code}")
        if response.status_code == 200:
            result = ExportTraceServiceResponse.FromString(response.content)
            if result.partial_success.rejected_spans:
                raise RuntimeError(f"OTLP rejected {result.partial_success.rejected_spans} span(s)")
        return response


class Telemetry:
    def __init__(
        self,
        settings: Settings,
        task_id: str,
        agent_version: str,
        *,
        resource_attributes: Mapping[str, Any] | None = None,
    ):
        identity = {
            "service.name": "rfi-agent",
            "hopper.task.id": task_id,
            "hopper.agent.version": agent_version,
        }
        additions = dict(resource_attributes or {})
        if identity.keys() & additions.keys():
            raise ValueError("Task telemetry resource identity cannot be replaced")
        self._exporter = _CheckedExporter(
            OTLPSpanExporter(
                endpoint=settings.endpoint.rstrip("/") + "/v1/traces",
                compression=Compression.Gzip,
                session=_TraceSession(),
            )
        )
        self._provider = TracerProvider(
            resource=Resource({**additions, **identity}), shutdown_on_exit=False
        )
        self._provider.add_span_processor(BatchSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("hopper.otel")
        self._is_closed = False

    @contextmanager
    def span(
        self,
        operation: str,
        name: str,
        *,
        conversation_id: str,
        attributes: Mapping[str, Any] | None = None,
        root: bool = False,
    ) -> Iterator[Span]:
        if self._is_closed:
            raise RuntimeError("Task telemetry is closed")
        identity_attribute = {
            "invoke_agent": "gen_ai.agent.name",
            "chat": "gen_ai.request.model",
            "execute_tool": "gen_ai.tool.name",
        }[operation]
        values = {
            **(attributes or {}),
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": "anthropic",
            "gen_ai.conversation.id": conversation_id,
            identity_attribute: name,
        }
        with self._tracer.start_as_current_span(
            f"{operation} {name}",
            context=Context() if root else None,
            kind=SpanKind.CLIENT if operation == "chat" else SpanKind.INTERNAL,
            attributes=values,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except BaseException as error:
                record_error(span, error)
                raise

    def close(self) -> None:
        if self._is_closed:
            return
        self._is_closed = True
        failure = None
        try:
            if not self._provider.force_flush():
                failure = RuntimeError("OTLP trace flush timed out")
        except Exception as error:
            failure = error
        try:
            self._provider.shutdown()
        except Exception as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"Telemetry shutdown also failed: {error}")
        if failure is None:
            failure = self._exporter.error
        if failure is not None:
            raise RuntimeError("Task telemetry could not be exported") from failure

    def __enter__(self) -> "Telemetry":
        return self

    def __exit__(self, exc_type, error, traceback) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Telemetry cleanup failed: {cleanup_error}")
