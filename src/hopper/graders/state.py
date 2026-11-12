import json
import math
import re
from dataclasses import dataclass
from datetime import datetime

from hopper.corpus import Corpus

MAX_RETRIEVED_CHARACTERS = 24_000
RETRIEVAL_TOOLS = frozenset({"search_documents", "read_section", "list_addenda", "read_drawing"})


@dataclass(frozen=True)
class _Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time: datetime
    operation: str | None
    tool_name: str | None
    is_successful: bool
    attributes: dict

    def error(self, attribute: str, message: str) -> ValueError:
        return ValueError(f"Trace {self.trace_id}, span {self.span_id}, {attribute}: {message}")


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _reject_constant(value)
    return number


def _decode(value: str, expected: type, context: str):
    try:
        decoded = json.loads(value, parse_constant=_reject_constant, parse_float=_finite_float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{context}: invalid JSON") from error
    if not isinstance(decoded, expected):
        raise ValueError(f"{context}: expected {expected.__name__}")
    return decoded


def _prepare(rows: list[dict]) -> list[_Span]:
    if len({row["trace_id"] for row in rows}) != 1:
        raise ValueError("rfi_core requires spans from exactly one trace")
    unique = {}
    for row in rows:
        context = f"Trace {row['trace_id']}, span {row['span_id']}"
        attributes = _decode(row["attributes"], dict, f"{context}, attributes")
        is_successful = row["status"] != "ERROR" and row["error_type"] is None
        keys = {}
        if row["parent_span_id"] is None:
            keys.update({"gen_ai.input.messages": list, "gen_ai.output.messages": list})
        if row["operation"] == "execute_tool":
            keys["gen_ai.tool.call.arguments"] = dict if is_successful else object
            if is_successful and row["tool_name"] in RETRIEVAL_TOOLS:
                keys["gen_ai.tool.call.result"] = list
        parsed = {
            key: _decode(attributes[key], expected, f"{context}, {key}")
            for key, expected in keys.items()
            if key in attributes
        }
        span = _Span(
            row["trace_id"],
            row["span_id"],
            row["parent_span_id"],
            row["start_time"],
            row["operation"],
            row["tool_name"],
            is_successful,
            parsed,
        )
        previous = unique.get(span.span_id)
        if previous is not None and (
            span != previous
            or json.dumps(span.attributes, sort_keys=True)
            != json.dumps(previous.attributes, sort_keys=True)
        ):
            raise span.error("duplicate", "conflicting span contents")
        unique.setdefault(span.span_id, span)
    return sorted(unique.values(), key=lambda span: (span.start_time, span.span_id))


def _message_texts(span: _Span, attribute: str, role: str) -> list[str]:
    texts = []
    for message in span.attributes.get(attribute, []):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise span.error(attribute, "message must have a role")
        if message["role"] != role:
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            raise span.error(attribute, "message parts must be a list")
        fragments = []
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                raise span.error(attribute, "message part must have a type")
            if part["type"] != "text":
                continue
            if not isinstance(part.get("content"), str):
                raise span.error(attribute, "text part content must be a string")
            fragments.append(part["content"])
        texts.append("".join(fragments))
    return texts


def _retrieved_text(span: _Span, corpus: Corpus) -> str:
    attribute = "gen_ai.tool.call.result"
    fragments = []
    for part in span.attributes.get(attribute, []):
        if not isinstance(part, dict):
            raise span.error(attribute, "retrieval content must contain objects")
        if part.get("type") == "text":
            if not isinstance(part.get("text"), str):
                raise span.error(attribute, "retrieved text must be a string")
            fragments.append(part["text"])
            continue
        uri = part.get("uri")
        if (
            part.get("type") != "uri"
            or span.tool_name != "read_drawing"
            or not isinstance(uri, str)
        ):
            raise span.error(attribute, "unsupported retrieval content")
        match = re.fullmatch(r"corpus://drawings/([A-Z]-[0-9]{3})\.png", uri)
        if match is None or match[1] not in corpus.drawings:
            raise span.error(attribute, f"unknown drawing reference: {uri}")
        fragments.append(corpus.documents[match[1]])
    return "\n\n".join(fragments)


def rfi_core(rows: list[dict], corpus: Corpus) -> dict:
    spans = _prepare(rows)
    roots = [span for span in spans if span.parent_span_id is None]
    if len(roots) > 1:
        raise ValueError(f"Trace {spans[0].trace_id} has multiple roots")
    tools = [span for span in spans if span.operation == "execute_tool"]
    submissions = [
        span for span in tools if span.tool_name == "submit_answer" and span.is_successful
    ]
    if len(submissions) > 1:
        raise ValueError(f"Trace {spans[0].trace_id} has multiple successful submissions")
    questions = _message_texts(roots[0], "gen_ai.input.messages", "user") if roots else []
    answer = None
    if submissions:
        answer = submissions[0].attributes.get("gen_ai.tool.call.arguments")
    elif roots:
        output = "".join(_message_texts(roots[0], "gen_ai.output.messages", "assistant"))
        if output:
            answer = _decode(
                output,
                dict,
                f"Trace {roots[0].trace_id}, span {roots[0].span_id}, final_answer",
            )
    needs_engineer = answer.get("needs_engineer") if answer is not None else None
    if needs_engineer is not None and type(needs_engineer) is not bool:
        raise ValueError(f"Trace {spans[0].trace_id}: needs_engineer must be a boolean")
    retrieved = [
        _retrieved_text(span, corpus)
        for span in tools
        if span.is_successful and span.tool_name in RETRIEVAL_TOOLS
    ]
    return {
        "rfi_question": questions[-1] if questions else None,
        "final_answer": answer,
        "retrieved_text": "\n\n".join(retrieved)[:MAX_RETRIEVED_CHARACTERS],
        "tool_calls": [
            {"name": span.tool_name, "arguments": span.attributes.get("gen_ai.tool.call.arguments")}
            for span in tools
        ],
        "needs_engineer": needs_engineer,
    }
