import json
from dataclasses import dataclass
from pathlib import Path

from hopper.checks import MODES
from hopper.tasks import load_tasks


def seed_ids(data_dir: Path) -> list[str]:
    from hopper.query import connect

    with connect(data_dir) as connection:
        return [
            row[0]
            for row in connection.execute(
                """SELECT DISTINCT json_extract_string(resource, '$."hopper.seed.id"') AS seed
               FROM spans WHERE service = 'rfi-agent' AND seed IS NOT NULL
               AND replay_of IS NULL AND eval_run_id IS NULL AND eval_case_id IS NULL
               ORDER BY seed"""
            ).fetchall()
        ]


def load_seed_tasks(dataset: Path) -> dict[str, dict]:
    tasks = {}
    for task in load_tasks(dataset):
        identity = task.get("id")
        if not isinstance(identity, str) or not identity or identity in tasks:
            raise ValueError(f"Missing or duplicate task ID in {dataset}: {identity}")
        if task.get("mode") not in MODES:
            raise ValueError(f"Invalid mode for task {identity}")
        tasks[identity] = task
    return tasks


@dataclass(frozen=True)
class SeedTrace:
    trace_id: str
    task_id: str
    conversation_id: str
    finished: bool
    spans: list[dict]


def read_seed_traces(connection, seed_id: str) -> list[SeedTrace]:
    if not seed_id.strip():
        raise ValueError("seed_id must be nonempty")
    cursor = connection.execute(
        """
        WITH selected AS (
            SELECT trace_id FROM spans GROUP BY trace_id
            HAVING bool_or(service = 'rfi-agent'
                AND json_extract_string(resource, '$."hopper.seed.id"') = ?)
            AND count(*) FILTER (WHERE replay_of IS NOT NULL
                OR eval_run_id IS NOT NULL OR eval_case_id IS NOT NULL) = 0
        )
        SELECT s.*, t.finished FROM spans s
        JOIN selected USING (trace_id) JOIN traces t USING (trace_id)
        ORDER BY s.trace_id, s.start_time, s.span_id
        """,
        [seed_id],
    )
    columns = [column[0] for column in cursor.description]
    grouped = {}
    for row in cursor.fetchall():
        span = dict(zip(columns, row, strict=True))
        span["attributes"] = json.loads(span["attributes"])
        span["resource"] = json.loads(span["resource"])
        if not isinstance(span["attributes"], dict) or not isinstance(span["resource"], dict):
            raise ValueError(f"Invalid attribute object in trace {span['trace_id']}")
        grouped.setdefault(span["trace_id"], []).append(span)
    traces = []
    for trace_id, spans in grouped.items():
        for key in ("task_id", "conversation_id", "agent_version"):
            values = {span[key] for span in spans}
            if len(values) != 1 or None in values or "" in values:
                raise ValueError(f"Missing or conflicting {key} in trace {trace_id}")
        if any(span["resource"].get("hopper.seed.id") != seed_id for span in spans):
            raise ValueError(f"Conflicting seed identity in trace {trace_id}")
        if len({span["span_id"] for span in spans}) != len(spans):
            raise ValueError(f"Duplicate span identity in trace {trace_id}")
        traces.append(
            SeedTrace(
                trace_id,
                spans[0]["task_id"],
                spans[0]["conversation_id"],
                spans[0]["finished"],
                spans,
            )
        )
    return traces


def submitted_answer(trace: SeedTrace) -> dict | None:
    submissions = [
        span
        for span in trace.spans
        if span["operation"] == "execute_tool"
        and span["tool_name"] == "submit_answer"
        and span["status"] != "ERROR"
        and span["error_type"] is None
    ]
    if len(submissions) > 1:
        raise ValueError(f"Multiple successful submissions in trace {trace.trace_id}")
    try:
        if submissions:
            answer = json.loads(submissions[0]["attributes"]["gen_ai.tool.call.arguments"])
            return answer if isinstance(answer, dict) else None
        roots = [span for span in trace.spans if span["parent_span_id"] is None]
        if len(roots) != 1:
            return None
        messages = json.loads(roots[0]["attributes"].get("gen_ai.output.messages", "[]"))
        texts = [
            part["content"]
            for message in messages
            if message["role"] == "assistant"
            for part in message["parts"]
            if part["type"] == "text"
        ]
        answer = json.loads("".join(texts))
        return answer if isinstance(answer, dict) else None
    except (ValueError, KeyError, TypeError):
        return None
