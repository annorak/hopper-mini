import fcntl
import json
import math
import os
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO

from hopper.config import Settings
from hopper.corpus import load_corpus
from hopper.graders import STATE_FIELDS, JevGrader
from hopper.graders.jev import hash_state
from hopper.graders.state import rfi_core
from hopper.query import connect

DEFAULT_LABELS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "labels"
_MAX_PLAN_SEARCH_STEPS = 100_000
type HumanAnswer = Literal["yes", "no"]


@dataclass(frozen=True)
class Label:
    grader: str
    grader_version: int
    question: str
    fields: tuple[str, ...]
    applies_to: str
    trace_id: str
    task_id: str | None
    checker_passed: bool
    labeled_at: str
    state: dict
    input_hash: str
    label: HumanAnswer


@dataclass(frozen=True)
class _TraceEvidence:
    trace_id: str
    task_id: str | None
    checker_passed: bool
    states: dict[str, dict]
    input_hashes: dict[str, str]


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _reject_constant(value)
    return number


def _parse_label(data: object, grader: JevGrader) -> Label:
    if not isinstance(data, dict) or data.keys() != Label.__dataclass_fields__.keys():
        raise ValueError("Label has missing or unknown fields")
    for name in ("grader", "question", "applies_to", "trace_id", "labeled_at"):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError(f"{name} must be nonempty text")
    if data["grader"] != grader.name:
        raise ValueError("Label grader does not match its file")
    if type(data["grader_version"]) is not int or data["grader_version"] < 1:
        raise ValueError("grader_version must be a positive integer")
    if data["task_id"] is not None and (
        not isinstance(data["task_id"], str) or not data["task_id"].strip()
    ):
        raise ValueError("task_id must be nonempty text or null")
    if type(data["checker_passed"]) is not bool:
        raise ValueError("checker_passed must be a boolean")
    fields = data["fields"]
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or field not in STATE_FIELDS for field in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("fields must be distinct rfi_core field names")
    if not isinstance(data["state"], dict) or data["state"].keys() != set(fields):
        raise ValueError("state must contain exactly the projected fields")
    if data["label"] not in ("yes", "no"):
        raise ValueError("label must be yes or no")
    timestamp = datetime.fromisoformat(data["labeled_at"])
    if timestamp.utcoffset() is None:
        raise ValueError("labeled_at must include a timezone")
    if (
        not isinstance(data["input_hash"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", data["input_hash"])
        or data["input_hash"] != hash_state(data["state"])
    ):
        raise ValueError("input_hash does not match saved state")
    if (
        data["question"] != grader.question
        or tuple(fields) != grader.fields
        or data["applies_to"] != grader.applies_to
    ):
        raise ValueError(
            "Incompatible question, fields, or applies_to; use the matching grader "
            "definition or a separate --labels-dir"
        )
    return Label(**{**data, "fields": tuple(fields)})


def _read_locked(file: TextIO, path: Path, grader: JevGrader) -> list[Label]:
    file.seek(0)
    try:
        content = file.read()
    except UnicodeError as error:
        raise ValueError(f"Invalid label file {path}: expected UTF-8") from error
    if content and not content.endswith("\n"):
        raise ValueError(
            f"Invalid label file {path}: unterminated final record; preserve this file "
            "and inspect or restore the interrupted record before retrying"
        )
    labels = {}
    for number, line in enumerate(content.split("\n")[:-1], 1):
        try:
            data = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
            label = _parse_label(data, grader)
            previous = labels.get(label.input_hash)
            if previous is not None and previous.label != label.label:
                raise ValueError("Conflicting duplicate human answers for the same evidence")
            labels.setdefault(label.input_hash, label)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"Invalid label file {path}, line {number}: {error}") from error
    return list(labels.values())


def read_labels(directory: Path, grader: JevGrader) -> list[Label]:
    path = directory / f"{grader.name}.jsonl"
    try:
        file = path.open(encoding="utf-8")
    except FileNotFoundError:
        return []
    with file:
        fcntl.flock(file, fcntl.LOCK_SH)
        return _read_locked(file, path, grader)


def _append_label(directory: Path, grader: JevGrader, label: Label) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{grader.name}.jsonl"
    document = asdict(label)
    document["fields"] = list(label.fields)
    _parse_label(document, grader)
    line = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        existing = {item.input_hash: item for item in _read_locked(file, path, grader)}
        previous = existing.get(label.input_hash)
        if previous is not None:
            if previous.label != label.label:
                raise ValueError(f"Conflicting duplicate human answers in {path}")
            return False
        file.write(line + "\n")
        file.flush()
        os.fsync(file.fileno())
        for sync_path in (directory, directory.parent):
            descriptor = os.open(sync_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return True


def _candidates(settings: Settings, graders: tuple[JevGrader, ...]) -> list[_TraceEvidence]:
    candidates = []
    services = {grader.applies_to for grader in graders} - {"*"}
    if len(services) > 1:
        raise ValueError("Selected graders have no shared applicable service")
    with connect(settings.data_dir) as connection:
        cursor = connection.execute(
            "SELECT t.trace_id, t.task_id, t.service, c.passed FROM traces t "
            "JOIN checks c ON c.trace_id = t.trace_id "
            "WHERE t.finished AND t.replay_of IS NULL AND t.eval_run_id IS NULL "
            "AND t.eval_case_id IS NULL AND (? OR t.service = ?) "
            "ORDER BY t.trace_id",
            [not services, next(iter(services), None)],
        )
        traces = cursor.fetchall()
        identities = [trace[0] for trace in traces]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate stored check for an eligible trace")
        if not traces:
            return candidates
        cursor = connection.execute(
            "SELECT * FROM spans WHERE trace_id IN (SELECT unnest(?))",
            [identities],
        )
        columns = [column[0] for column in cursor.description]
        spans = {trace_id: [] for trace_id in identities}
        for row in cursor.fetchall():
            span = dict(zip(columns, row, strict=True))
            spans[span["trace_id"]].append(span)
        corpus = load_corpus()
        random.shuffle(traces)
        for trace_id, task_id, _service, checker_passed in traces:
            if type(checker_passed) is not bool:
                raise ValueError(f"Stored check for trace {trace_id} must have a boolean passed")
            state = rfi_core(spans[trace_id], corpus)
            projections = {
                grader.name: {field: state[field] for field in grader.fields} for grader in graders
            }
            hashes = {name: hash_state(projected) for name, projected in projections.items()}
            candidates.append(
                _TraceEvidence(trace_id, task_id, checker_passed, projections, hashes)
            )
    return candidates


def _has_capacity(available: dict[bool, set[str]], needed: dict[bool, int]) -> bool:
    return (
        len(available[False]) >= needed[False]
        and len(available[True]) >= needed[True]
        and len(available[False] | available[True]) >= sum(needed.values())
    )


def _can_finish(
    candidates: list[_TraceEvidence], known: dict[str, set[str]], needed: dict[str, dict[bool, int]]
) -> bool:
    for name, quota in needed.items():
        available = {False: set(), True: set()}
        for candidate in candidates:
            digest = candidate.input_hashes[name]
            if digest not in known[name]:
                available[candidate.checker_passed].add(digest)
        if not _has_capacity(available, quota):
            return False
    return True


def _plan(
    graders: tuple[JevGrader, ...],
    candidates: list[_TraceEvidence],
    saved: dict[str, list[Label]],
    n: int,
) -> list[tuple[_TraceEvidence, tuple[JevGrader, ...]]]:
    known = {name: {label.input_hash for label in labels} for name, labels in saved.items()}
    needed = {}
    for grader in graders:
        labels = saved[grader.name]
        remaining = max(0, n - len(labels))
        failures = sum(not label.checker_passed for label in labels)
        need_failure = min(remaining, max(0, (n + 1) // 2 - failures))
        needed[grader.name] = {False: need_failure, True: remaining - need_failure}
    saved_traces = {}
    for name, labels in saved.items():
        for label in labels:
            saved_traces.setdefault(label.trace_id, {})[name] = label
    partial_traces = {
        trace_id: labels for trace_id, labels in saved_traces.items() if len(labels) < len(graders)
    }
    by_trace = {candidate.trace_id: candidate for candidate in candidates}
    plan = []
    for trace_id, labels in partial_traces.items():
        candidate = by_trace.get(trace_id)
        if candidate is None or any(
            candidate.input_hashes[name] != label.input_hash
            or candidate.checker_passed != label.checker_passed
            for name, label in labels.items()
        ):
            raise ValueError(
                f"Cannot resume incomplete cohort trace {trace_id}: original projected evidence "
                "and checker result must still be available"
            )
        passed = candidate.checker_passed
        missing = tuple(grader for grader in graders if grader.name not in labels)
        if any(
            not needed[grader.name][passed]
            or candidate.input_hashes[grader.name] in known[grader.name]
            for grader in missing
        ):
            raise ValueError(
                f"Cannot resume incomplete cohort trace {trace_id}: its missing labels conflict "
                "with saved evidence or the required checker-failure/checker-pass balance"
            )
        for grader in missing:
            known[grader.name].add(candidate.input_hashes[grader.name])
            needed[grader.name][passed] -= 1
        plan.append((candidate, missing))
    candidates = [candidate for candidate in candidates if candidate.trace_id not in partial_traces]
    if not _can_finish(candidates, known, needed):
        raise ValueError(
            f"Cannot reach {n} labels per grader: insufficient unique projected states "
            "in the checker-failure/checker-pass pools; failure/pass evidence must be disjoint"
        )
    next_index = 0
    history = []
    search_steps = 0
    while any(sum(quota.values()) for quota in needed.values()):
        for index in range(next_index, len(candidates)):
            search_steps += 1
            if search_steps > _MAX_PLAN_SEARCH_STEPS:
                raise ValueError(
                    f"Shared cohort search limit reached after {_MAX_PLAN_SEARCH_STEPS} "
                    "candidate attempts; feasibility is unknown"
                )
            candidate = candidates[index]
            passed = candidate.checker_passed
            missing = tuple(
                grader
                for grader in graders
                if needed[grader.name][passed]
                and candidate.input_hashes[grader.name] not in known[grader.name]
            )
            unfinished = tuple(grader for grader in graders if sum(needed[grader.name].values()))
            if not missing or missing != unfinished:
                continue
            for grader in missing:
                known[grader.name].add(candidate.input_hashes[grader.name])
                needed[grader.name][passed] -= 1
            if _can_finish(candidates[index + 1 :], known, needed):
                plan.append((candidate, missing))
                history.append(index + 1)
                next_index = index + 1
                break
            for grader in missing:
                known[grader.name].remove(candidate.input_hashes[grader.name])
                needed[grader.name][passed] += 1
        else:
            if not history:
                raise ValueError(
                    f"Could not form a shared {n}-trace cohort with unique projected evidence "
                    "for every grader and the required checker-failure/checker-pass balance"
                )
            next_index = history.pop()
            candidate, missing = plan.pop()
            for grader in missing:
                known[grader.name].remove(candidate.input_hashes[grader.name])
                needed[grader.name][candidate.checker_passed] += 1
    return plan


def label_traces(
    settings: Settings,
    graders: Sequence[JevGrader],
    n: int,
    labels_dir: Path,
    ask: Callable[[str], HumanAnswer],
    show: Callable[[str], None],
) -> dict[str, int]:
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive target label count")
    graders = tuple(graders)
    if not graders or any(not isinstance(grader, JevGrader) for grader in graders):
        raise ValueError("Labeling requires Jev graders")
    if len({grader.name for grader in graders}) != len(graders):
        raise ValueError("Labeling requires distinct grader names")
    saved = {grader.name: read_labels(labels_dir, grader) for grader in graders}
    pending = tuple(grader for grader in graders if len(saved[grader.name]) < n)
    if not pending:
        return {name: len(labels) for name, labels in saved.items()}
    candidates = _candidates(settings, graders)
    plan = _plan(graders, candidates, saved, n)
    for evidence, missing in plan:
        for grader in missing:
            state = evidence.states[grader.name]
            input_hash = evidence.input_hashes[grader.name]
            current = read_labels(labels_dir, grader)
            if len(current) >= n or any(label.input_hash == input_hash for label in current):
                continue
            show(
                f"Trace: {evidence.trace_id}\nTask: {evidence.task_id}\n"
                f"Grader: {grader.name} v{grader.version}\n"
                + json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
            )
            answer = ask(grader.question)
            if answer not in ("yes", "no"):
                raise ValueError("Human answer must be explicit yes or no")
            label = Label(
                grader=grader.name,
                grader_version=grader.version,
                question=grader.question,
                fields=grader.fields,
                applies_to=grader.applies_to,
                trace_id=evidence.trace_id,
                task_id=evidence.task_id,
                checker_passed=evidence.checker_passed,
                labeled_at=datetime.now(UTC).isoformat(),
                state=state,
                input_hash=input_hash,
                label=answer,
            )
            _append_label(labels_dir, grader, label)
    return {grader.name: len(read_labels(labels_dir, grader)) for grader in graders}
