import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hopper.query import connect
from hopper.seed import SeedTrace, load_seed_tasks, read_seed_traces, submitted_answer


@dataclass(frozen=True)
class Metric:
    name: str
    observed: str
    target: str
    passed: bool | None


@dataclass(frozen=True)
class Profile:
    seed_id: str
    finished: int
    pending: int
    span_count: int
    metrics: list[Metric]
    agent_versions: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.finished > 0 and all(metric.passed is not False for metric in self.metrics)

    def render(self) -> str:
        lines = [
            f"Seed {self.seed_id}: {self.finished} finished traces, {self.pending} pending, "
            f"{self.span_count} spans",
            f"Agent versions: {', '.join(self.agent_versions) or 'none'}",
            "metric\tobserved\ttarget\tresult",
        ]
        for metric in self.metrics:
            status = "REPORT" if metric.passed is None else "PASS" if metric.passed else "MISS"
            lines.append(f"{metric.name}\t{metric.observed}\t{metric.target}\t{status}")
        return "\n".join(lines)


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _depth(trace: SeedTrace) -> int:
    parents = {span["span_id"]: span["parent_span_id"] for span in trace.spans}
    if sum(parent is None for parent in parents.values()) != 1:
        return 0
    maximum = 0
    for identity in parents:
        visited = set()
        while identity is not None:
            if identity in visited or identity not in parents:
                return 0
            visited.add(identity)
            identity = parents[identity]
        maximum = max(maximum, len(visited))
    return maximum


def _bytes(value) -> bytes:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return text.encode("utf-8")


def _has_image(value) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "uri" and value.get("modality") == "image" and value.get("uri"):
            return True
        return any(_has_image(item) for item in value.values())
    return isinstance(value, list) and any(_has_image(item) for item in value)


def profile_seed(data_dir: Path, dataset: Path, seed_id: str = "seed") -> Profile:
    modes = {identity: task["mode"] for identity, task in load_seed_tasks(dataset).items()}
    with connect(data_dir) as connection:
        selected = read_seed_traces(connection, seed_id)
    traces = [trace for trace in selected if trace.finished]
    unknown = {trace.task_id for trace in traces} - modes.keys()
    if unknown:
        raise ValueError(f"Unknown seed task IDs: {', '.join(sorted(unknown))}")
    spans = [span for trace in traces for span in trace.spans]
    counts = [len(trace.spans) for trace in traces]
    depths = [_depth(trace) for trace in traces]
    sizes = [len(_bytes(value)) for span in spans for value in span["attributes"].values()]
    kinds = dict.fromkeys(
        (
            "JSON tool arguments",
            "Markdown spec text",
            "plain-text answers",
            "Python code",
            "image URI parts",
        ),
        False,
    )
    sent_bytes, unique = 0, set()
    for trace in traces:
        answer = submitted_answer(trace)
        kinds["plain-text answers"] |= bool(
            answer and isinstance(answer.get("text"), str) and answer["text"].strip()
        )
    for span in spans:
        attrs = span["attributes"]
        if span["operation"] == "execute_tool":
            try:
                arguments = json.loads(attrs.get("gen_ai.tool.call.arguments", "null"))
            except json.JSONDecodeError:
                arguments = None
            kinds["JSON tool arguments"] |= isinstance(arguments, dict)
            if span["tool_name"] == "calculate" and isinstance(arguments, dict):
                kinds["Python code"] |= bool(
                    isinstance(arguments.get("code"), str) and arguments["code"].strip()
                )
            if span["tool_name"] == "read_section" and span["status"] != "ERROR":
                result = json.loads(attrs.get("gen_ai.tool.call.result", "[]"))
                kinds["Markdown spec text"] |= any(
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and part.get("text", "").startswith("#")
                    and "SPEC-" in part["text"]
                    for part in result
                )
        for name in (
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.tool.call.result",
        ):
            content = json.loads(attrs.get(name, "[]"))
            kinds["image URI parts"] |= _has_image(content)
            if span["operation"] != "chat" or name != "gen_ai.input.messages":
                continue
            for message in content:
                encoded = _bytes(message)
                sent_bytes += len(encoded)
                unique.add((span["conversation_id"], encoded))
    tools = [span for span in spans if span["operation"] == "execute_tool"]
    errors = [span for span in tools if span["status"] == "ERROR"]
    error_types = {span["error_type"] for span in errors if span["error_type"]}
    keys = {key for span in spans for key in span["attributes"]}
    conversations = Counter(trace.conversation_id for trace in traces)
    multiple = sum(count > 1 for count in conversations.values())
    calculations = [trace for trace in traces if modes[trace.task_id] == "calculation"]
    calculated_ids = {
        trace.trace_id
        for trace in calculations
        if any(
            span["operation"] == "execute_tool" and span["tool_name"] == "calculate"
            for span in trace.spans
        )
    }
    base_calculations = [
        trace
        for trace in calculations
        if any(
            span["parent_span_id"] is None and span["attributes"].get("hopper.turn") == 0
            for span in trace.spans
        )
    ]
    calculated = len(calculated_ids)
    base_calculated = sum(trace.trace_id in calculated_ids for trace in base_calculations)
    median, p95 = _percentile(counts, 0.5), _percentile(counts, 0.95)
    p50, p99 = _percentile(sizes, 0.5), _percentile(sizes, 0.99)
    error_rate = len(errors) / len(tools) if tools else math.nan
    conversation_rate = multiple / len(conversations) if conversations else math.nan
    calculation_rate = calculated / len(calculations) if calculations else math.nan
    unique_bytes = sum(len(encoded) for _, encoded in unique)
    metrics = [
        Metric("Spans per trace median", f"{median:g}", ">= 12", median >= 12),
        Metric("Spans per trace p95", f"{p95:g}", ">= 25", p95 >= 25),
        Metric(
            "Tree depth",
            f"min={min(depths, default=0)}; below 3={sum(d < 3 for d in depths)}; "
            f"invalid={depths.count(0)}",
            "every trace >= 3",
            bool(depths) and min(depths) >= 3,
        ),
        *[
            Metric(f"Content: {name}", "present" if present else "absent", "present", present)
            for name, present in kinds.items()
        ],
        Metric("Attribute value p50 bytes", f"{p50:g}", "<= 4096", p50 <= 4096),
        Metric("Attribute value p99 bytes", f"{p99:g}", ">= 30720", p99 >= 30720),
        Metric(
            "Tool error rate",
            f"{len(errors)}/{len(tools)} ({error_rate:.2%})",
            "2% to 10%",
            0.02 <= error_rate <= 0.1,
        ),
        Metric(
            "Tool error types",
            f"{len(error_types)}: {', '.join(sorted(error_types))}",
            ">= 3",
            len(error_types) >= 3,
        ),
        Metric("Span attribute keys", str(len(keys)), ">= 25", len(keys) >= 25),
        Metric(
            "Multi-trace conversations",
            f"{multiple}/{len(conversations)} ({conversation_rate:.2%})",
            ">= 15%",
            conversation_rate >= 0.15,
        ),
        Metric(
            "Calculation traces using calculate",
            f"all turns={calculated}/{len(calculations)} ({calculation_rate:.2%}); "
            f"base RFIs={base_calculated}/{len(base_calculations)}",
            ">= 80%",
            calculation_rate >= 0.8,
        ),
        Metric(
            "Chat input message bytes",
            f"sent={sent_bytes}; once per conversation={unique_bytes}",
            "reported; captured URI messages, canonical JSON, no array framing",
            None,
        ),
    ]
    return Profile(
        seed_id,
        len(traces),
        len(selected) - len(traces),
        len(spans),
        metrics,
        tuple(sorted({span["agent_version"] for span in spans})),
    )
