import fcntl
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml

STATE_FIELDS = frozenset(
    {"rfi_question", "final_answer", "retrieved_text", "tool_calls", "needs_engineer"}
)
Status = Literal["draft", "active", "retired"]
Origin = Literal["seed", "human", "discovered"]


@dataclass(frozen=True)
class ThresholdValidation:
    no_threshold: float
    yes_threshold: float
    agreement: float | None
    precision_no: float | None
    recall_no: float | None


@dataclass(frozen=True)
class Validation:
    labels: int
    agreement: float | None
    precision_no: float | None = None
    recall_no: float | None = None
    threshold_sweep: tuple[ThresholdValidation, ...] = ()
    validated_at: str | None = None
    definition_hash: str | None = None
    labels_hash: str | None = None


@dataclass(frozen=True, kw_only=True)
class Definition:
    name: str
    version: int
    status: Status
    applies_to: str
    fields: tuple[str, ...]
    origin: Origin
    validation: Validation | None = None


@dataclass(frozen=True, kw_only=True)
class JevGrader(Definition):
    question: str
    model: str
    yes_threshold: float
    no_threshold: float
    kind: Literal["jev"] = field(default="jev", init=False)


@dataclass(frozen=True, kw_only=True)
class CodeGrader(Definition):
    function: str
    kind: Literal["code"] = field(default="code", init=False)


type Grader = JevGrader | CodeGrader


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _name(value: object) -> str:
    name = _text(value, "name")
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name):
        raise ValueError("Grader name must be lowercase snake_case")
    return name


def _version(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Grader version must be a positive integer")
    return value


def _probability(value: object, name: str) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _parse_metrics(document: dict, labels: int) -> dict[str, float | None]:
    values = {}
    for name in ("agreement", "precision_no", "recall_no"):
        value = document[name]
        values[name] = None if value is None else _probability(value, name)
        if labels == 0 and value is not None:
            raise ValueError("Validation metrics must be null when labels is zero")
    if labels > 0 and values["agreement"] is None:
        raise ValueError("Validation agreement cannot be null when labels is positive")
    return values


def _parse_validation(document: dict) -> Validation:
    legacy = {"labels", "agreement"}
    full = legacy | {
        "precision_no",
        "recall_no",
        "threshold_sweep",
        "validated_at",
        "definition_hash",
        "labels_hash",
    }
    if document.keys() not in (legacy, full):
        raise ValueError("Validation has missing or unknown fields")
    labels = document["labels"]
    if type(labels) is not int or labels < 0:
        raise ValueError("Validation labels must be a nonnegative integer")
    if document.keys() == legacy:
        return Validation(labels, _probability(document["agreement"], "agreement"))
    data = dict(document, **_parse_metrics(document, labels))
    validated_at = _text(data["validated_at"], "validated_at")
    try:
        timestamp = datetime.fromisoformat(validated_at)
    except ValueError as error:
        raise ValueError("validated_at must be an ISO timestamp with a timezone") from error
    if timestamp.utcoffset() is None:
        raise ValueError("validated_at must be an ISO timestamp with a timezone")
    for name in ("definition_hash", "labels_hash"):
        if not re.fullmatch(r"[0-9a-f]{64}", _text(data[name], name)):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    sweep = data["threshold_sweep"]
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("threshold_sweep must be a nonempty list")
    rows = []
    pairs = set()
    for row in sweep:
        if not isinstance(row, dict) or row.keys() != {
            "no_threshold",
            "yes_threshold",
            "agreement",
            "precision_no",
            "recall_no",
        }:
            raise ValueError("Threshold validation has missing or unknown fields")
        no = _probability(row["no_threshold"], "no_threshold")
        yes = _probability(row["yes_threshold"], "yes_threshold")
        if no >= yes or (no, yes) in pairs:
            raise ValueError("Threshold pairs must be distinct with no_threshold < yes_threshold")
        pairs.add((no, yes))
        rows.append(ThresholdValidation(no, yes, **_parse_metrics(row, labels)))
    data["threshold_sweep"] = tuple(rows)
    return Validation(**data)


def definition_fingerprint(grader: JevGrader) -> str:
    document = asdict(grader)
    del document["status"], document["validation"]
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def threshold_pairs(grader: JevGrader) -> tuple[tuple[float, float], ...]:
    pairs = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    current = (grader.no_threshold, grader.yes_threshold)
    if current not in pairs:
        pairs.append(current)
    return tuple(pairs)


def _check_validation(grader: JevGrader, report: Validation) -> None:
    if report.definition_hash != definition_fingerprint(grader):
        raise ValueError("Validation requires a full report bound to the current grader definition")
    document = asdict(report)
    document["threshold_sweep"] = list(document["threshold_sweep"])
    report = _parse_validation(document)
    rows = {(row.no_threshold, row.yes_threshold): row for row in report.threshold_sweep}
    if rows.keys() != set(threshold_pairs(grader)):
        raise ValueError("Validation threshold sweep must include all required threshold pairs")
    current = rows[(grader.no_threshold, grader.yes_threshold)]
    if (report.agreement, report.precision_no, report.recall_no) != (
        current.agreement,
        current.precision_no,
        current.recall_no,
    ):
        raise ValueError("Validation metrics must match the configured threshold row")


def validate_model(value: object) -> str:
    model = _text(value, "model")
    if not re.fullmatch(r"jev-[0-9]+\.[0-9]+\.[0-9]+", model):
        raise ValueError("Jev model must be a pinned version such as jev-1.14.0")
    return model


def _parse(document: object, resolve_model: Callable[[], str] | None = None) -> Grader:
    if not isinstance(document, dict):
        raise ValueError("Grader definition must be a mapping")
    data = dict(document)
    kind = data.pop("kind", None)
    if kind not in ("jev", "code"):
        raise ValueError("Grader kind must be jev or code")
    if kind == "jev" and resolve_model is not None:
        data.setdefault("model", None)
    common = {"name", "version", "status", "applies_to", "fields", "origin"}
    specific = (
        {"question", "model", "yes_threshold", "no_threshold"} if kind == "jev" else {"function"}
    )
    required = common | specific
    if required - data.keys() or data.keys() - required - {"validation"}:
        raise ValueError("Grader definition has missing or unknown fields")
    data["name"] = _name(data["name"])
    data["version"] = _version(data["version"])
    if data["status"] not in ("draft", "active", "retired"):
        raise ValueError("Grader status must be draft, active, or retired")
    if data["origin"] not in ("seed", "human", "discovered"):
        raise ValueError("Grader origin must be seed, human, or discovered")
    data["applies_to"] = _text(data["applies_to"], "applies_to")
    fields = data["fields"]
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(item, str) or item not in STATE_FIELDS for item in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("fields must be a nonempty list of distinct rfi_core field names")
    data["fields"] = tuple(fields)
    validation = data.get("validation")
    if validation is not None:
        if kind != "jev" or not isinstance(validation, dict):
            raise ValueError("Only Jev graders accept label validation")
        data["validation"] = _parse_validation(validation)
    if kind == "code":
        function = _text(data["function"], "function")
        if not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", function):
            raise ValueError("Code function must use module:function syntax")
        return CodeGrader(**data)
    data["question"] = _text(data["question"], "question")
    data["yes_threshold"] = _probability(data["yes_threshold"], "yes_threshold")
    data["no_threshold"] = _probability(data["no_threshold"], "no_threshold")
    if data["no_threshold"] >= data["yes_threshold"]:
        raise ValueError("no_threshold must be less than yes_threshold")
    if data["model"] in (None, "jev-latest") and resolve_model is not None:
        data["model"] = resolve_model()
    data["model"] = validate_model(data["model"])
    return JevGrader(**data)


class _DefinitionLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Grader mapping keys must be strings")
            if key in keys:
                raise ValueError(f"Duplicate YAML key: {key}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def read_document(path: Path) -> dict:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_DefinitionLoader)
        if not isinstance(document, dict):
            raise ValueError("Grader definition must be a mapping")
        return document
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid grader definition {path}: {error}") from error


def read_definition(path: Path) -> Grader:
    try:
        grader = _parse(read_document(path))
        if path.parent.name != grader.name or path.name != f"v{grader.version}.yaml":
            raise ValueError("Name and version do not match the grader file path")
        return grader
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid grader definition {path}: {error}") from error


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


SEED_QUESTIONS = {
    "answered_question": (
        "`final_answer` directly answers `rfi_question`.",
        ("rfi_question", "final_answer"),
    ),
    "grounded": (
        "Every factual claim in `final_answer` is supported by `retrieved_text`.",
        ("final_answer", "retrieved_text"),
    ),
    "actionable": (
        "A general contractor could act on `final_answer` without asking a follow-up question.",
        ("rfi_question", "final_answer"),
    ),
}


def _seed_jev(name: str, model: str) -> JevGrader:
    question, fields = SEED_QUESTIONS[name]
    return JevGrader(
        name=name,
        version=1,
        status="active",
        applies_to="rfi-agent",
        fields=fields,
        origin="seed",
        question=question,
        model=model,
        yes_threshold=0.7,
        no_threshold=0.3,
    )


SEED_CODE = CodeGrader(
    name="cites_sources",
    version=1,
    status="active",
    applies_to="rfi-agent",
    fields=("final_answer",),
    origin="seed",
    function="hopper.graders.code:cites_sources",
)


class Catalog:
    def __init__(self, data_dir: Path):
        self.directory = data_dir / "graders"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with (self.directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _path(self, name: str, version: int) -> Path:
        return self.directory / _name(name) / f"v{_version(version)}.yaml"

    def _write(self, grader: Grader) -> None:
        path = self._path(grader.name, grader.version)
        path.parent.mkdir(exist_ok=True)
        temporary = path.parent / f".v{grader.version}-{uuid4()}.tmp"
        document = asdict(grader)
        if grader.validation is not None and grader.validation.definition_hash is None:
            document["validation"] = {
                "labels": grader.validation.labels,
                "agreement": grader.validation.agreement,
            }
        with temporary.open("x", encoding="utf-8") as output:
            yaml.safe_dump(document, output, sort_keys=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
        _sync_directory(self.directory)

    def list_graders(self) -> list[Grader]:
        with self._locked():
            graders = [read_definition(path) for path in self.directory.glob("*/v*.yaml")]
        return sorted(graders, key=lambda grader: (grader.name, grader.version))

    def create_draft(
        self, document: dict, *, resolve_model: Callable[[], str] | None = None
    ) -> Grader:
        if {"version", "status", "validation"} & document.keys():
            raise ValueError("New graders cannot specify version, status, or validation")
        name = _name(document.get("name"))
        with self._locked():
            versions = [
                read_definition(path).version for path in (self.directory / name).glob("v*.yaml")
            ]
            grader = _parse(
                dict(document, version=max(versions, default=0) + 1, status="draft"),
                resolve_model,
            )
            self._write(grader)
        return grader

    def promote(
        self,
        name: str,
        version: int,
        *,
        run_code_tests: Callable[[CodeGrader], None],
    ) -> Grader:
        from hopper.query import connect, validate_grader_names

        with self._locked():
            grader = read_definition(self._path(name, version))
            if grader.status != "draft":
                raise ValueError("Only draft graders can be promoted")
            with connect(self.directory.parent) as connection:
                validate_grader_names(connection, [grader.name])
            if isinstance(grader, JevGrader):
                validation = grader.validation
                if (
                    validation is None
                    or validation.labels < 50
                    or validation.agreement is None
                    or validation.agreement < 0.85
                ):
                    raise ValueError("Promotion requires at least 50 labels and 0.85 agreement")
                _check_validation(grader, validation)
            else:
                run_code_tests(grader)
            promoted = replace(grader, status="active")
            self._write(promoted)
        return promoted

    def save_validation(self, expected_grader: JevGrader, report: Validation) -> JevGrader:
        _check_validation(expected_grader, report)
        with self._locked():
            current = read_definition(self._path(expected_grader.name, expected_grader.version))
            if not isinstance(current, JevGrader) or definition_fingerprint(
                current
            ) != definition_fingerprint(expected_grader):
                raise ValueError("Grader definition changed during validation; validate it again")
            validated = replace(current, validation=report)
            self._write(validated)
        return validated

    def retire(self, name: str, version: int) -> Grader:
        with self._locked():
            grader = read_definition(self._path(name, version))
            if grader.status == "retired":
                raise ValueError("A retired grader cannot transition again")
            retired = replace(grader, status="retired")
            self._write(retired)
        return retired

    def install_seeds(self, resolve_model: Callable[[], str]) -> list[Grader]:
        names = (*SEED_QUESTIONS, SEED_CODE.name)
        with self._locked():
            existing = {
                name: read_definition(self._path(name, 1))
                for name in names
                if self._path(name, 1).exists()
            }
            for name, grader in existing.items():
                if name == SEED_CODE.name:
                    expected = SEED_CODE
                elif isinstance(grader, JevGrader):
                    expected = _seed_jev(name, grader.model)
                else:
                    raise ValueError(f"Existing {name} v1 conflicts with its seed definition")
                if replace(grader, status="active", validation=None) != expected:
                    raise ValueError(f"Existing {name} v1 conflicts with its seed definition")
            missing_jev = [name for name in SEED_QUESTIONS if name not in existing]
            if missing_jev:
                models = {
                    grader.model for grader in existing.values() if isinstance(grader, JevGrader)
                }
                if len(models) > 1:
                    raise ValueError("Partial seed installation has inconsistent model pins")
                model = next(iter(models)) if models else validate_model(resolve_model())
                for name in missing_jev:
                    self._write(_seed_jev(name, model))
            if SEED_CODE.name not in existing:
                self._write(SEED_CODE)
            graders = [read_definition(self._path(name, 1)) for name in names]
            for grader in graders:
                _sync_directory(self._path(grader.name, 1).parent)
            _sync_directory(self.directory)
        return graders

    def retire_superseded(self, has_coverage: Callable[[Grader], bool]) -> list[Grader]:
        retired = []
        with self._locked():
            active = [
                grader
                for path in self.directory.glob("*/v*.yaml")
                if (grader := read_definition(path)).status == "active"
            ]
            for name in sorted({grader.name for grader in active}):
                versions = sorted(
                    (grader for grader in active if grader.name == name),
                    key=lambda grader: grader.version,
                )
                if len(versions) < 2 or not has_coverage(versions[-1]):
                    continue
                for grader in versions[:-1]:
                    replacement = replace(grader, status="retired")
                    self._write(replacement)
                    retired.append(replacement)
        return retired
