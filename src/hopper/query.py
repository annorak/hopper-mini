"""Read published Parquet files without opening the writer or WAL."""

import json
import re
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import yaml

from hopper.writer import JSON_COLUMNS, SCHEMA

TRACE_PROJECTION = """
trace_id,
coalesce(any_value(service) FILTER (WHERE parent_span_id IS NULL),
    CASE WHEN count(DISTINCT service) = 1 THEN min(service) END) AS service,
any_value(name) FILTER (WHERE parent_span_id IS NULL) AS root_name,
min(start_time) AS start_time,
max(end_time) AS end_time,
date_diff('millisecond', min(start_time), max(end_time)) AS duration_ms,
count(*) AS span_count,
count(*) FILTER (WHERE status = 'ERROR') AS error_count,
count(*) FILTER (WHERE operation = 'chat') AS llm_calls,
count(*) FILTER (WHERE operation = 'execute_tool') AS tool_calls,
sum(input_tokens) AS input_tokens,
sum(output_tokens) AS output_tokens,
any_value(conversation_id) AS conversation_id,
any_value(task_id) AS task_id,
any_value(agent_version) AS agent_version,
any_value(replay_of) AS replay_of,
any_value(eval_run_id) AS eval_run_id,
any_value(eval_case_id) AS eval_case_id,
bool_or(parent_span_id IS NULL) AS has_root,
min(received_at) AS first_received_at,
max(received_at) AS last_received_at,
max(log_offset) AS last_log_offset,
(bool_or(parent_span_id IS NULL) AND max(received_at) < now() - INTERVAL 10 SECOND)
    OR max(received_at) < now() - INTERVAL 10 MINUTE AS finished
"""
GRADE_COLUMNS = {
    "trace_id": "VARCHAR",
    "grader": "VARCHAR",
    "grader_version": "INTEGER",
    "label": "VARCHAR",
    "p_yes": "DOUBLE",
    "reason": "VARCHAR",
    "kind": "VARCHAR",
    "source": "VARCHAR",
    "model": "VARCHAR",
    "input_hash": "VARCHAR",
    "cost_usd": "DOUBLE",
    "created_at": "TIMESTAMPTZ",
    "trace_log_offset": "BIGINT",
    "seed_log_offset": "BIGINT",
}
CHECK_COLUMNS = {
    "trace_id": "VARCHAR",
    "task_id": "VARCHAR",
    "passed": "BOOLEAN",
    "criteria": "JSON",
    "created_at": "TIMESTAMPTZ",
}


@dataclass(frozen=True)
class GradeFilter:
    grader: str
    label: str


@dataclass(frozen=True)
class FindFilters:
    start: datetime | None = None
    end: datetime | None = None
    service: str | None = None
    status: str | None = None
    tool: str | None = None
    text: str | None = None
    grades: tuple[GradeFilter, ...] = ()
    task: str | None = None
    conversation: str | None = None
    limit: int = 100


@dataclass(frozen=True)
class QueryProfile:
    total_time: float
    total_bytes_read: int
    total_row_groups_scanned: int
    total_row_groups_to_scan: int
    total_rows_scanned: int

    def render(self) -> str:
        return (
            f"wall time: {self.total_time:.6f}s; bytes read: {self.total_bytes_read}; "
            f"row groups scanned: {self.total_row_groups_scanned}/"
            f"{self.total_row_groups_to_scan}; rows scanned: {self.total_rows_scanned}"
        )


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: list[tuple]
    profile: QueryProfile | None = None


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _empty_table(connection, name, columns):
    fields = ", ".join(f"{_identifier(key)} {kind}" for key, kind in columns.items())
    connection.execute(f"CREATE TABLE {name} ({fields})")


def _active_graders(data_dir: Path) -> list[tuple[str, int, str]]:
    active = []
    for path in sorted((data_dir / "graders").glob("*/v*.yaml")):
        definition = yaml.safe_load(path.read_text())
        match = re.fullmatch(r"v([1-9][0-9]*)\.yaml", path.name)
        if not isinstance(definition, dict) or match is None:
            raise ValueError(f"Invalid grader definition: {path}")
        version = definition.get("version")
        if (
            definition.get("name") != path.parent.name
            or type(version) is not int
            or version != int(match[1])
            or definition.get("status") not in {"draft", "active", "retired"}
        ):
            raise ValueError(f"Invalid grader name, version, or status: {path}")
        if definition["status"] == "active":
            applies_to = definition.get("applies_to")
            if not isinstance(applies_to, str) or not applies_to.strip():
                raise ValueError(f"Invalid grader applies_to: {path}")
            active.append((path.parent.name, version, applies_to))
    return active


def validate_grader_names(connection, names: list[str]) -> None:
    trace_columns = {
        column[0].casefold()
        for column in connection.execute("SELECT * FROM traces LIMIT 0").description
    }
    folded_names = [name.casefold() for name in names]
    if len(set(folded_names)) != len(names) or trace_columns.intersection(folded_names):
        raise ValueError("Active grader names collide with trace columns or each other")


def _grade_projection(names: list[str]) -> str:
    if not names:
        return "SELECT t.* FROM selected t"
    columns = ", ".join(
        f"max(label) FILTER (WHERE grader = {_literal(name)}) AS {_identifier(name)}"
        for name in names
    )
    return (
        f"SELECT t.*, wide.* EXCLUDE (trace_id) FROM selected t LEFT JOIN "
        f"(SELECT trace_id, {columns} FROM grades GROUP BY trace_id) wide USING (trace_id)"
    )


def _span_snapshot_files(data_dir: Path) -> list[str]:
    files = []
    ranges = []
    for path in (data_dir / "spans").glob("*.parquet"):
        match = re.fullmatch(r"([0-9]+)-([0-9]+)\.parquet", path.name)
        if match is None:
            files.append(str(path))
        else:
            start, end = map(int, match.groups())
            if end <= start:
                raise ValueError(f"Invalid published span range: {path.name}")
            ranges.append((start, end, path))
    position = 0
    for start, end, path in sorted(ranges):
        if start < position:
            raise ValueError(f"Overlapping published span range: {path.name}")
        if start > position:
            break
        files.append(str(path))
        position = end
    return sorted(files)


@contextmanager
def connect(data_dir: Path):
    """Open one UTC query snapshot over complete published files, including empty stores."""
    connection = duckdb.connect(config={"threads": 2, "read_ahead_depth": 0})
    try:
        connection.execute("SET TimeZone = 'UTC'")
        files = _span_snapshot_files(Path(data_dir))
        if files:
            connection.read_parquet(files, hive_partitioning=False).create_view("_span_files")
        else:
            connection.register("_span_files", pa.Table.from_pylist([], schema=SCHEMA))
        casts = ", ".join(f"{name}::JSON AS {name}" for name in JSON_COLUMNS)
        connection.execute(f"CREATE VIEW spans AS SELECT * REPLACE ({casts}) FROM _span_files")
        connection.execute(
            f"CREATE VIEW traces AS SELECT {TRACE_PROJECTION} FROM spans GROUP BY trace_id"
        )

        partitions = []
        for index, folder in enumerate(sorted((Path(data_dir) / "grades").glob("grader=*"))):
            files = sorted(str(path) for path in folder.glob("*.parquet"))
            if not files:
                continue
            relation = f"_grade_{index}"
            rows = connection.read_parquet(files, hive_partitioning=False, union_by_name=True)
            missing = {"trace_log_offset", "seed_log_offset"} - set(rows.columns)
            rows.create_view(relation)
            grader = folder.name.removeprefix("grader=")
            expressions = {name: _identifier(name) for name in GRADE_COLUMNS}
            expressions.update({name: "NULL" for name in missing}, grader=_literal(grader))
            fields = ", ".join(
                f"{expressions[name]}::{kind} AS {_identifier(name)}"
                for name, kind in GRADE_COLUMNS.items()
            )
            partitions.append(f"SELECT {fields} FROM {relation}")
        if partitions:
            connection.execute("CREATE VIEW grades_all AS " + " UNION ALL ".join(partitions))
        else:
            _empty_table(connection, "grades_all", GRADE_COLUMNS)
        connection.execute(
            "CREATE TABLE active_graders "
            "(grader VARCHAR, grader_version INTEGER, applies_to VARCHAR)"
        )
        active = _active_graders(Path(data_dir))
        if active:
            connection.executemany("INSERT INTO active_graders VALUES (?, ?, ?)", active)
        connection.execute("""
            CREATE VIEW _fresh_grade_versions AS SELECT g.* FROM grades_all g
            JOIN traces t ON t.trace_id = g.trace_id
            LEFT JOIN traces seed ON seed.trace_id = t.replay_of
            WHERE g.trace_log_offset = t.last_log_offset
              AND ((t.replay_of IS NULL AND g.seed_log_offset IS NULL)
                OR (t.replay_of IS NOT NULL AND g.seed_log_offset = seed.last_log_offset))
            QUALIFY row_number() OVER (
                PARTITION BY g.trace_id, g.grader, g.grader_version
                ORDER BY g.created_at DESC) = 1
        """)
        connection.execute("""
            CREATE VIEW grades AS SELECT g.* FROM _fresh_grade_versions g
            JOIN active_graders a USING (grader, grader_version)
            JOIN traces t ON t.trace_id = g.trace_id
            WHERE a.applies_to = '*' OR a.applies_to = t.service
            QUALIFY row_number() OVER (PARTITION BY g.trace_id, g.grader
                ORDER BY g.grader_version DESC, g.created_at DESC) = 1
        """)
        names = sorted({name for name, _, _ in active})
        validate_grader_names(connection, names)
        connection.execute(
            "CREATE VIEW trace_grades AS WITH selected AS (SELECT * FROM traces) "
            + _grade_projection(names)
        )

        files = sorted(str(path) for path in (Path(data_dir) / "checks").glob("*.parquet"))
        if files:
            connection.read_parquet(files, hive_partitioning=False).create_view("_check_files")
            fields = ", ".join(
                f"{_identifier(name)}::{kind} AS {_identifier(name)}"
                for name, kind in CHECK_COLUMNS.items()
            )
            connection.execute(f"CREATE VIEW checks AS SELECT {fields} FROM _check_files")
        else:
            _empty_table(connection, "checks", CHECK_COLUMNS)
        yield connection
    finally:
        connection.close()


def _execute(connection, sql: str, parameters: dict, profile: bool) -> QueryResult:
    with tempfile.TemporaryDirectory(prefix="hopper-query-") as temporary:
        path = Path(temporary) / "profile.json"
        if profile:
            connection.execute("PRAGMA enable_profiling = 'json'")
            connection.execute("SET profiling_output = ?", [str(path)])
            connection.execute("SET profiling_coverage = 'ALL'")
        metrics = None
        try:
            cursor = connection.execute(sql, parameters)
            columns = tuple(column[0] for column in cursor.description)
            rows = cursor.fetchall()
            if profile:
                try:
                    contents = json.loads(path.read_text())
                    query = contents["query"]
                    metrics = QueryProfile(
                        total_time=query["total_time"],
                        total_bytes_read=contents["io"]["total_bytes_read"],
                        total_row_groups_scanned=query["total_row_groups_scanned"],
                        total_row_groups_to_scan=query["total_row_groups_to_scan"],
                        total_rows_scanned=query["total_rows_scanned"],
                    )
                except (OSError, KeyError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        "DuckDB did not produce the required JSON profile metrics"
                    ) from error
        finally:
            if profile:
                connection.execute("PRAGMA disable_profiling")
        return QueryResult(columns, rows, metrics)


def execute_sql(
    data_dir: Path, sql: str, parameters: dict | None = None, *, profile: bool = False
) -> QueryResult:
    with connect(data_dir) as connection:
        return _execute(connection, sql, parameters or {}, profile)


def find_traces(data_dir: Path, filters: FindFilters, *, profile: bool = False) -> QueryResult:
    if filters.limit <= 0:
        raise ValueError("--limit must be positive")
    if filters.start and filters.end and filters.start >= filters.end:
        raise ValueError("--from must be earlier than --to")
    parameters = {"limit": filters.limit}
    having = []
    where = []
    if filters.start is not None:
        parameters["start"] = filters.start
        where.append("start_time >= $start")
    root = ["parent_span_id IS NULL"]
    if filters.start is not None:
        root.append("start_time >= $start")
    if filters.end is not None:
        parameters["end"] = filters.end
        root.append("start_time < $end")
    if filters.start is not None or filters.end is not None:
        having.append("bool_or(" + " AND ".join(root) + ")")
    for name, column in (("status", "status"), ("tool", "tool_name")):
        value = getattr(filters, name)
        if value is not None:
            parameters[name] = value
            having.append(f"bool_or({column} = ${name})")
    if filters.text is not None:
        parameters["text"] = filters.text
        having.append("bool_or(contains(lower(CAST(attributes AS VARCHAR)), lower($text)))")
    predicates = []
    for name, column in (
        ("service", "service"),
        ("task", "task_id"),
        ("conversation", "conversation_id"),
    ):
        value = getattr(filters, name)
        if value is not None:
            parameters[name] = value
            predicates.append(f"t.{column} = ${name}")
    for index, grade in enumerate(filters.grades):
        if not grade.grader or grade.label not in {"yes", "no", "unsure"}:
            raise ValueError("--grade must be grader=yes, grader=no, or grader=unsure")
        parameters[f"grader_{index}"] = grade.grader
        parameters[f"label_{index}"] = grade.label
        predicates.append(
            "EXISTS (SELECT 1 FROM grades g WHERE g.trace_id = t.trace_id "
            f"AND g.grader = $grader_{index} AND g.label = $label_{index})"
        )
    aggregate = f"SELECT {TRACE_PROJECTION} FROM spans"
    if where:
        aggregate += " WHERE " + " AND ".join(where)
    aggregate += " GROUP BY trace_id"
    if having:
        aggregate += " HAVING " + " AND ".join(having)
    selected = "SELECT * FROM aggregated t"
    if predicates:
        selected += " WHERE " + " AND ".join(predicates)
    with connect(data_dir) as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT grader FROM active_graders ORDER BY grader"
            ).fetchall()
        ]
        sql = (
            f"WITH aggregated AS ({aggregate}), selected AS ({selected}) "
            + _grade_projection(names)
            + " ORDER BY start_time DESC, trace_id LIMIT $limit"
        )
        return _execute(connection, sql, parameters, profile)


def read_trace(data_dir: Path, trace_id: str) -> QueryResult:
    return execute_sql(
        data_dir,
        "SELECT * FROM spans WHERE trace_id = $trace_id ORDER BY start_time, span_id",
        {"trace_id": trace_id},
    )


def parse_window(since: str | None, start: str | None, end: str | None):
    """Parse a half-open UTC window; naive ISO timestamps also mean UTC."""
    if since is not None and (start is not None or end is not None):
        raise ValueError("--since cannot be combined with --from or --to")
    if since is not None:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhdw])", since)
        if match is None or float(match[1]) <= 0:
            raise ValueError("--since must be a positive duration such as 1h or 7d")
        seconds = float(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[match[2]]
        try:
            return datetime.now(UTC) - timedelta(seconds=seconds), None
        except OverflowError as error:
            raise ValueError("--since duration is too large") from error
    bounds = []
    for flag, value in (("--from", start), ("--to", end)):
        if value is None:
            bounds.append(None)
            continue
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{flag} must be an ISO timestamp") from error
        bounds.append(
            parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
        )
    if bounds[0] is not None and bounds[1] is not None and bounds[0] >= bounds[1]:
        raise ValueError("--from must be earlier than --to")
    return tuple(bounds)


def parse_grade(value: str) -> GradeFilter:
    grader, separator, label = value.partition("=")
    if not separator or not grader or label not in {"yes", "no", "unsure"}:
        raise ValueError("--grade must be grader=yes, grader=no, or grader=unsure")
    return GradeFilter(grader, label)


def _messages(value):
    if not isinstance(value, str):
        return [json.dumps(value, ensure_ascii=False)]
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if not isinstance(decoded, list):
        return [value]
    return [
        message if isinstance(message, str) else json.dumps(message, ensure_ascii=False)
        for message in decoded
    ]


def render_trace(result: QueryResult, *, full: bool = False) -> str:
    """Render every span once, including orphans, cycles, and disconnected components."""
    spans = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
    children = defaultdict(list)
    ids = {span["span_id"] for span in spans}
    for index, span in enumerate(spans):
        children[span["parent_span_id"]].append(index)
    roots = [
        index
        for index, span in enumerate(spans)
        if span["parent_span_id"] is None or span["parent_span_id"] not in ids
    ]
    visited = set()
    lines = []
    stack = [(index, 0) for index in reversed(roots + list(range(len(spans))))]
    while stack:
        index, depth = stack.pop()
        if index in visited:
            continue
        visited.add(index)
        span = spans[index]
        prefix = "  " * depth
        lines.append(
            f"{prefix}{span['span_id']} {span['name']} "
            f"start={span['start_time']} duration={span['duration_ms']}ms "
            f"operation={span['operation']} model={span['model']} "
            f"tokens={span['input_tokens']}/{span['output_tokens']} "
            f"tool={span['tool_name']} status={span['status']}"
        )
        attributes = json.loads(span["attributes"])
        for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
            if key not in attributes:
                continue
            for message in _messages(attributes[key]):
                preview = message if full or len(message) <= 200 else message[:200] + "..."
                lines.append(f"{prefix}  {key}: {preview}")
        stack.extend((child, depth + 1) for child in reversed(children[span["span_id"]]))
    return "\n".join(lines)
