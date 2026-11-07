import fcntl
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

MODES = (
    "lookup",
    "multi_section",
    "missed_addendum",
    "conflict",
    "needs_engineer",
    "calculation",
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _number(value) -> Decimal | None:
    if type(value) not in (int, float):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def check_answer(key: dict, answer: dict) -> dict:
    """Evaluate only the section 4.1 criteria applicable to this answer key."""
    key = key if isinstance(key, dict) else {}
    answer = answer if isinstance(answer, dict) else {}
    mode = key.get("mode")
    text = answer.get("text")
    citations = answer.get("citations")
    has_citations = isinstance(citations, list) and all(
        isinstance(citation, str) for citation in citations
    )
    cited = set(citations) if has_citations else set()
    criteria = {}
    if mode != "needs_engineer":
        required = key.get("required_citations")
        criteria["citations"] = (
            has_citations
            and isinstance(required, list)
            and all(isinstance(citation, str) and citation in cited for citation in required)
        )
    phrases = key.get("required_phrases")
    if phrases:
        criteria["phrases"] = (
            isinstance(text, str)
            and isinstance(phrases, list)
            and all(
                isinstance(phrase, str) and _normalize(phrase) in _normalize(text)
                for phrase in phrases
            )
        )
    if mode == "missed_addendum":
        citation = key.get("addendum_citation")
        criteria["addendum_cited"] = isinstance(citation, str) and citation in cited
    if mode == "conflict":
        documents = key.get("conflict_documents")
        criteria["both_sides_named"] = (
            isinstance(text, str)
            and isinstance(documents, list)
            and len(documents) == 2
            and all(
                isinstance(document, str)
                and re.search(rf"(?<![\w-]){re.escape(document)}(?![\w-])", text) is not None
                for document in documents
            )
        )
        citation = key.get("precedence_citation")
        criteria["precedence_cited"] = isinstance(citation, str) and citation in cited
    criteria["escalation"] = (
        mode in MODES
        and type(answer.get("needs_engineer")) is bool
        and type(key.get("needs_engineer")) is bool
        and answer["needs_engineer"] == key["needs_engineer"]
    )
    if mode == "calculation":
        value = _number(answer.get("value"))
        expected = _number(key.get("value"))
        tolerance = _number(key.get("tolerance"))
        criteria["value_in_tolerance"] = (
            value is not None
            and expected is not None
            and tolerance is not None
            and tolerance >= 0
            and abs(value - expected) <= tolerance
        )
    return {"pass": all(criteria.values()), "criteria": criteria}


@dataclass(frozen=True)
class CheckSummary:
    seed_id: str
    written: int
    unchanged: int
    missing_answers: int
    pending: int
    modes: dict[str, list[bool]]
    criteria: dict[str, list[bool]]

    def render(self) -> str:
        results = [value for values in self.modes.values() for value in values]
        lines = [
            f"Seed {self.seed_id}: checked={len(results)} written={self.written} "
            f"unchanged={self.unchanged} pending={self.pending} "
            f"missing_answers={self.missing_answers}",
            "group\tname\tpassed\ttotal\tpass_rate",
        ]
        for group, groups in (
            ("overall", {"all": results}),
            ("mode", self.modes),
            ("criterion", self.criteria),
        ):
            for name, values in groups.items():
                rate = f"{sum(values) / len(values):.2%}" if values else "n/a"
                lines.append(f"{group}\t{name}\t{sum(values)}\t{len(values)}\t{rate}")
        return "\n".join(lines)


def check_seed(data_dir: Path, dataset: Path, seed_id: str = "seed") -> CheckSummary:
    from hopper.seed import load_seed_tasks

    tasks = load_seed_tasks(dataset)
    for identity, task in tasks.items():
        if (
            not isinstance(task.get("answer_key"), dict)
            or task["answer_key"].get("mode") != task["mode"]
        ):
            raise ValueError(f"Answer-key mode differs for task {identity}")
    for folder in (data_dir, data_dir / "tmp", data_dir / "checks"):
        folder.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(data_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    with (data_dir / ".checks.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _check_seed_locked(data_dir, tasks, seed_id)


def _check_seed_locked(data_dir: Path, tasks: dict[str, dict], seed_id: str) -> CheckSummary:
    from hopper.query import connect
    from hopper.seed import read_seed_traces, submitted_answer

    with connect(data_dir) as connection:
        selected = read_seed_traces(connection, seed_id)
        traces = [trace for trace in selected if trace.finished]
        unknown = {trace.task_id for trace in traces} - tasks.keys()
        if unknown:
            raise ValueError(f"Unknown seed task IDs: {', '.join(sorted(unknown))}")
        existing = {}
        for trace_id, task_id, passed, criteria in connection.execute(
            "SELECT trace_id, task_id, passed, criteria FROM checks"
        ).fetchall():
            if trace_id in existing:
                raise ValueError(f"Duplicate stored check for trace {trace_id}")
            existing[trace_id] = (task_id, passed, json.loads(criteria))
        modes = {mode: [] for mode in MODES}
        criteria_rates = defaultdict(list)
        rows, missing = [], 0
        for trace in traces:
            task = tasks[trace.task_id]
            answer = submitted_answer(trace)
            missing += answer is None
            result = check_answer(task["answer_key"], answer or {})
            current = (trace.task_id, result["pass"], result["criteria"])
            if trace.trace_id in existing and existing[trace.trace_id] != current:
                raise ValueError(
                    f"Stored check changed for trace {trace.trace_id}; "
                    "late spans or a changed answer key require reconciliation"
                )
            modes[task["mode"]].append(result["pass"])
            for name, passed in result["criteria"].items():
                criteria_rates[name].append(passed)
            if trace.trace_id not in existing:
                rows.append(
                    (
                        trace.trace_id,
                        trace.task_id,
                        result["pass"],
                        json.dumps(result["criteria"]),
                        datetime.now(UTC),
                    )
                )
        if rows:
            connection.execute("CREATE TEMP TABLE pending_checks AS SELECT * FROM checks LIMIT 0")
            connection.executemany("INSERT INTO pending_checks VALUES (?, ?, ?, ?, ?)", rows)
            identity = uuid4().hex
            temporary = data_dir / "tmp" / f"checks-{identity}.parquet"
            final = data_dir / "checks" / f"{identity}.parquet"
            connection.execute(
                "COPY pending_checks TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)]
            )
            with temporary.open("rb") as output:
                os.fsync(output.fileno())
            os.rename(temporary, final)
        descriptor = os.open(data_dir / "checks", os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return CheckSummary(
        seed_id,
        len(rows),
        len(traces) - len(rows),
        missing,
        len(selected) - len(traces),
        modes,
        dict(sorted(criteria_rates.items())),
    )
