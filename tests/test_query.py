import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from hopper.cli import app
from hopper.query import (
    CHECK_COLUMNS,
    FindFilters,
    GradeFilter,
    QueryResult,
    connect,
    execute_sql,
    find_traces,
    parse_grade,
    parse_window,
    read_trace,
    render_trace,
)
from hopper.writer import JSON_COLUMNS, SCHEMA

runner = CliRunner()


def span(trace, identity, start, *, parent=None, seconds=1, **values):
    row = dict.fromkeys(SCHEMA.names)
    row.update(
        trace_id=trace,
        span_id=identity,
        parent_span_id=parent,
        name=identity,
        service="rfi-agent",
        start_time=start,
        end_time=start + timedelta(seconds=seconds),
        duration_ms=seconds * 1000,
        received_at=start,
        status="OK",
        operation="chat",
        conversation_id="conversation",
        task_id="task",
        agent_version="v1",
        log_offset=0,
        attributes="{}",
        events="[]",
        links="[]",
        resource="{}",
    )
    row.update(values)
    return row


def write_spans(directory, rows, *, row_group_size=None):
    folder = directory / "spans"
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA),
        folder / "0-1.parquet",
        row_group_size=row_group_size,
    )


def grader(directory, name, version=1, status="active"):
    folder = directory / "graders" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"v{version}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "version": version,
                "status": status,
            }
        )
    )


def write_grades(directory, name, values):
    schema = pa.schema(
        [
            ("trace_id", pa.string()),
            ("grader", pa.string()),
            ("grader_version", pa.int32()),
            ("label", pa.string()),
            ("p_yes", pa.float64()),
            ("reason", pa.string()),
            ("kind", pa.string()),
            ("source", pa.string()),
            ("model", pa.string()),
            ("input_hash", pa.string()),
            ("cost_usd", pa.float64()),
            ("created_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    rows = []
    for trace, version, label, created in values:
        rows.append(
            dict(
                trace_id=trace,
                grader="wrong-physical-value",
                grader_version=version,
                label=label,
                p_yes=1.0 if label == "yes" else 0.0,
                reason=None,
                kind="code",
                source="live",
                model=None,
                input_hash="hash",
                cost_usd=0,
                created_at=created,
            )
        )
    folder = directory / "grades" / f"grader={name}"
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / "grades.parquet")


def dictionaries(result):
    return [dict(zip(result.columns, row, strict=True)) for row in result.rows]


@pytest.mark.parametrize("progress", [False, True])
def test_empty_relations_and_progress_files(tmp_path, progress):
    directory = tmp_path / "missing"
    if progress:
        write_spans(directory, [])
    with connect(directory) as connection:
        for name in (
            "spans",
            "traces",
            "grades_all",
            "active_graders",
            "grades",
            "trace_grades",
            "checks",
        ):
            assert connection.execute(f"SELECT * FROM {name}").fetchall() == []
        assert connection.execute("DESCRIBE spans").fetchall() == [
            (
                field.name,
                "JSON"
                if field.name in JSON_COLUMNS
                else (
                    "VARCHAR"
                    if pa.types.is_string(field.type)
                    else "TIMESTAMP WITH TIME ZONE"
                    if pa.types.is_timestamp(field.type)
                    else "DOUBLE"
                    if pa.types.is_floating(field.type)
                    else "BIGINT"
                ),
                "YES",
                None,
                None,
                None,
            )
            for field in SCHEMA
        ]
        assert [(row[0], row[1]) for row in connection.execute("DESCRIBE checks").fetchall()] == [
            (name, "TIMESTAMP WITH TIME ZONE" if kind == "TIMESTAMPTZ" else kind)
            for name, kind in CHECK_COLUMNS.items()
        ]
    assert find_traces(directory, FindFilters()).rows == []
    assert read_trace(directory, "missing").rows == []
    if not progress:
        assert not directory.exists()


def test_only_published_files_and_checks_are_independent(tmp_path):
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    (temporary / "incomplete.parquet").write_text("not parquet")
    folder = tmp_path / "checks"
    folder.mkdir()
    pq.write_table(
        pa.table(
            {
                "trace_id": ["checked"],
                "task_id": ["task"],
                "passed": [False],
                "criteria": ['{"answer": false}'],
                "created_at": [datetime.now(UTC)],
            }
        ),
        folder / "checks.parquet",
    )
    result = execute_sql(tmp_path, "SELECT trace_id, passed, criteria::VARCHAR FROM checks")
    assert result.rows == [("checked", False, '{"answer": false}')]
    assert find_traces(tmp_path, FindFilters()).rows == []


def test_exact_aggregate_and_finished_thresholds(tmp_path):
    now = datetime.now(UTC)
    rows = [
        span(
            "complete",
            "root",
            now,
            input_tokens=11,
            output_tokens=7,
            replay_of="seed",
            eval_run_id="run",
            eval_case_id="case",
        ),
        span(
            "complete",
            "child",
            now + timedelta(seconds=2),
            parent="root",
            seconds=3,
            service="child-service",
            status="ERROR",
            operation="execute_tool",
            input_tokens=13,
            output_tokens=17,
        ),
        span("null", "null-root", now),
        span("root-boundary", "boundary", now),
        span("root-old", "old", now),
        span("orphan-boundary", "orphan", now, parent="missing"),
        span("orphan-old", "orphan-old", now, parent="missing"),
    ]
    write_spans(tmp_path, rows)
    with connect(tmp_path) as connection:
        connection.execute("BEGIN")
        connection.execute("""
          CREATE OR REPLACE VIEW spans AS SELECT * REPLACE (
            CASE trace_id WHEN 'root-boundary' THEN now() - INTERVAL 10 SECOND
              WHEN 'orphan-boundary' THEN now() - INTERVAL 10 MINUTE
              WHEN 'orphan-old' THEN now() - INTERVAL '10 MINUTE 1 MICROSECOND'
              ELSE now() - INTERVAL '10 SECOND 1 MICROSECOND' END AS received_at
          ) FROM _span_files
        """)
        result = connection.execute("SELECT * FROM traces").fetchall()
        columns = tuple(column[0] for column in connection.description)
        traces = {row["trace_id"]: row for row in dictionaries(QueryResult(columns, result))}
    complete = traces["complete"]
    assert complete == {
        "trace_id": "complete",
        "service": "rfi-agent",
        "root_name": "root",
        "start_time": now,
        "end_time": now + timedelta(seconds=5),
        "duration_ms": 5000,
        "span_count": 2,
        "error_count": 1,
        "llm_calls": 1,
        "tool_calls": 1,
        "input_tokens": 24,
        "output_tokens": 24,
        "conversation_id": "conversation",
        "task_id": "task",
        "agent_version": "v1",
        "replay_of": "seed",
        "eval_run_id": "run",
        "eval_case_id": "case",
        "has_root": True,
        "first_received_at": complete["first_received_at"],
        "last_received_at": complete["last_received_at"],
        "finished": True,
    }
    assert complete["first_received_at"] == complete["last_received_at"]
    assert traces["null"]["input_tokens"] is None
    assert traces["null"]["output_tokens"] is None
    assert {
        name: traces[name]["finished"]
        for name in ("root-boundary", "root-old", "orphan-boundary", "orphan-old")
    } == {"root-boundary": False, "root-old": True, "orphan-boundary": False, "orphan-old": True}


def test_time_window_keeps_complete_trace_and_independent_span_matches(tmp_path):
    start = datetime(2026, 11, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    rows = [
        span("kept", "root", start, input_tokens=10),
        span("kept", "error", end, parent="root", status="ERROR", input_tokens=20),
        span("kept", "tool", end + timedelta(seconds=1), parent="root", tool_name="read"),
        span(
            "kept",
            "text",
            end + timedelta(seconds=2),
            parent="root",
            attributes=json.dumps({"text": "Literal %_ MATCH"}),
        ),
        span("old", "old-root", start - timedelta(seconds=1)),
        span(
            "old",
            "recent-child",
            start,
            parent="old-root",
            status="ERROR",
            tool_name="read",
            attributes=json.dumps({"text": "Literal %_ MATCH"}),
        ),
        span(
            "upper",
            "upper-root",
            end,
            status="ERROR",
            tool_name="read",
            attributes=json.dumps({"text": "Literal %_ MATCH"}),
        ),
        span(
            "orphan",
            "orphan",
            start,
            parent="unknown",
            status="ERROR",
            tool_name="read",
            attributes=json.dumps({"text": "Literal %_ MATCH"}),
        ),
    ]
    write_spans(tmp_path, rows)
    result = find_traces(
        tmp_path,
        FindFilters(
            start=start,
            end=end,
            status="ERROR",
            tool="read",
            text="literal %_ match",
            service="rfi-agent",
            task="task",
            conversation="conversation",
        ),
    )
    assert [row["trace_id"] for row in dictionaries(result)] == ["kept"]
    kept = dictionaries(result)[0]
    assert (kept["span_count"], kept["input_tokens"], kept["error_count"], kept["end_time"]) == (
        4,
        30,
        1,
        end + timedelta(seconds=3),
    )
    assert dictionaries(find_traces(tmp_path, FindFilters(end=end)))[0]["trace_id"] == "kept"
    assert find_traces(tmp_path, FindFilters(text="Literal _% MATCH")).rows == []


def test_current_active_grades_folder_ownership_and_null(tmp_path):
    now = datetime.now(UTC)
    write_spans(tmp_path, [span(trace, trace, now) for trace in ("a", "b", "c", "d")])
    for version, status in ((1, "active"), (2, "active"), (3, "draft"), (4, "retired")):
        grader(tmp_path, "answered_question", version, status)
    grader(tmp_path, "grounded")
    grader(tmp_path, "empty")
    write_grades(
        tmp_path,
        "answered_question",
        [
            ("a", 1, "yes", now + timedelta(seconds=4)),
            ("a", 2, "yes", now),
            ("a", 2, "no", now + timedelta(seconds=1)),
            ("a", 3, "yes", now + timedelta(seconds=2)),
            ("a", 4, "yes", now + timedelta(seconds=3)),
            ("b", 1, "yes", now),
        ],
    )
    write_grades(tmp_path, "grounded", [("b", 1, "no", now)])
    assert execute_sql(tmp_path, "SELECT DISTINCT grader FROM grades_all ORDER BY grader").rows == [
        ("answered_question",),
        ("grounded",),
    ]
    assert execute_sql(
        tmp_path,
        "SELECT trace_id, grader_version, label FROM grades "
        "WHERE grader = 'answered_question' ORDER BY trace_id",
    ).rows == [
        ("a", 2, "no"),
        ("b", 1, "yes"),
    ]
    result = find_traces(
        tmp_path,
        FindFilters(
            grades=(GradeFilter("answered_question", "yes"), GradeFilter("grounded", "no"))
        ),
    )
    assert [row["trace_id"] for row in dictionaries(result)] == ["b"]
    assert result.columns[-3:] == ("answered_question", "empty", "grounded")
    missing = dictionaries(find_traces(tmp_path, FindFilters()))[-1]
    assert missing["trace_id"] == "d"
    assert missing["answered_question"] is None
    assert missing["grounded"] is None
    assert missing["empty"] is None


def test_quoted_grader_and_hostile_values_are_bound(tmp_path):
    now = datetime.now(UTC)
    name = "grader'\"quote"
    trace = "x' OR TRUE --"
    write_spans(
        tmp_path,
        [
            span(
                trace,
                "root",
                now,
                service="service'quote",
                attributes=json.dumps({"text": "needle' OR TRUE --"}),
            )
        ],
    )
    grader(tmp_path, name)
    write_grades(tmp_path, name, [(trace, 1, "yes", now)])
    assert read_trace(tmp_path, trace).rows[0][0] == trace
    assert read_trace(tmp_path, "' OR TRUE --").rows == []
    result = find_traces(
        tmp_path,
        FindFilters(
            service="service'quote", text="needle' OR TRUE --", grades=(GradeFilter(name, "yes"),)
        ),
    )
    assert dictionaries(result)[0][name] == "yes"
    assert find_traces(tmp_path, FindFilters(service="' OR TRUE --")).rows == []


@pytest.mark.parametrize(
    "definition",
    [
        {"name": "wrong", "version": 1, "status": "active"},
        {"name": "test", "version": True, "status": "active"},
        {"name": "test", "version": 2, "status": "active"},
        {"name": "test", "version": 1, "status": "unknown"},
        "invalid",
        None,
    ],
)
def test_invalid_stored_grader_is_visible(tmp_path, definition):
    folder = tmp_path / "graders" / "test"
    folder.mkdir(parents=True)
    (folder / "v1.yaml").write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="Invalid grader"):
        execute_sql(tmp_path, "SELECT * FROM grades")


@pytest.mark.parametrize("names", [("TRACE_ID",), ("same", "SAME")])
def test_grade_column_collision_is_visible(tmp_path, names, monkeypatch):
    monkeypatch.setattr("hopper.query._active_graders", lambda _: [(name, 1) for name in names])
    with pytest.raises(ValueError, match="collide"):
        execute_sql(tmp_path, "SELECT * FROM trace_grades")


def test_render_each_message_preview_full_and_malformed_parent_graph(tmp_path):
    now = datetime.now(UTC)
    messages = ["first:" + "x" * 220, "second:" + "y" * 230]
    output = "output:" + "z" * 240
    attributes = json.dumps(
        {"gen_ai.input.messages": json.dumps(messages), "gen_ai.output.messages": output}
    )
    write_spans(
        tmp_path,
        [
            span(
                "t",
                "root",
                now,
                attributes=attributes,
                model="test-model",
                input_tokens=11,
                output_tokens=13,
                tool_name="test-tool",
            ),
            span("t", "child", now, parent="root"),
            span("t", "orphan", now, parent="missing"),
            span("t", "cycle-a", now, parent="cycle-b"),
            span("t", "cycle-b", now, parent="cycle-a"),
        ],
    )
    result = read_trace(tmp_path, "t")
    preview = render_trace(result)
    for message in messages + [output]:
        assert message[:200] + "..." in preview
        assert message[:201] not in preview
        assert message in render_trace(result, full=True)
    for identity in ("root", "child", "orphan", "cycle-a", "cycle-b"):
        assert sum(line.lstrip().startswith(identity + " ") for line in preview.splitlines()) == 1
    for value in (
        "operation=chat",
        "model=test-model",
        "tokens=11/13",
        "tool=test-tool",
        "status=OK",
        "duration=1000.0ms",
        "start=",
    ):
        assert value in preview
    assert "\n  child " in preview


@pytest.mark.parametrize(
    "since,start,end",
    [
        ("1h", "2026-11-01", None),
        ("1h", None, "2026-11-01"),
        ("0h", None, None),
        ("-1h", None, None),
        ("tomorrow", None, None),
        (None, "bad", None),
        (None, None, "bad"),
        (None, "2026-11-02", "2026-11-01"),
    ],
)
def test_invalid_window(since, start, end):
    with pytest.raises(ValueError):
        parse_window(since, start, end)


def test_utc_windows_and_grade_validation():
    assert parse_window(None, "2026-11-01", "2026-11-02T01:00:00+01:00") == (
        datetime(2026, 11, 1, tzinfo=UTC),
        datetime(2026, 11, 2, tzinfo=UTC),
    )
    lower, upper = parse_window("1h", None, None)
    assert abs((datetime.now(UTC) - lower).total_seconds() - 3600) < 1
    assert upper is None
    assert parse_grade("grounded=unsure") == GradeFilter("grounded", "unsure")
    for value in ("grounded", "=yes", "grounded=bad", "grounded=yes=no"):
        with pytest.raises(ValueError, match="--grade"):
            parse_grade(value)


def test_genuine_profile_prunes_old_row_groups(tmp_path):
    now = datetime.now(UTC)
    rows = [span(f"old-{index}", f"old-{index}", now - timedelta(days=20)) for index in range(100)]
    rows += [
        span("recent", "root", now - timedelta(minutes=5), input_tokens=7),
        span(
            "recent",
            "error",
            now - timedelta(minutes=4),
            parent="root",
            status="ERROR",
            input_tokens=11,
        ),
    ]
    write_spans(tmp_path, rows, row_group_size=5)
    result = find_traces(
        tmp_path, FindFilters(start=now - timedelta(hours=1), status="ERROR"), profile=True
    )
    assert [
        (row["trace_id"], row["span_count"], row["input_tokens"]) for row in dictionaries(result)
    ] == [("recent", 2, 18)]
    profile = result.profile
    assert 0 < profile.total_row_groups_scanned < profile.total_row_groups_to_scan
    assert profile.total_rows_scanned > 0
    assert profile.total_time > 0
    assert profile.total_bytes_read >= 0
    assert execute_sql(tmp_path, "CREATE TABLE local_table(i INTEGER)", profile=True).profile


@pytest.fixture
def canonical_store(tmp_path):
    now = datetime.now(UTC).replace(microsecond=0)
    recent = now - timedelta(minutes=30)
    rows = [
        span(
            "a",
            "a-root",
            recent,
            input_tokens=10,
            output_tokens=2,
            attributes=json.dumps({"text": "BUTTERFLY VALVE ADD-2"}),
        ),
        span(
            "a",
            "a-tool",
            recent + timedelta(seconds=1),
            parent="a-root",
            status="ERROR",
            tool_name="read_drawing",
            operation="execute_tool",
        ),
        span(
            "a",
            "a-chat",
            recent + timedelta(seconds=2),
            parent="a-root",
            seconds=2,
            input_tokens=20,
            output_tokens=3,
        ),
        span(
            "b",
            "b-root",
            now - timedelta(hours=2),
            seconds=2,
            service="other",
            input_tokens=5,
            output_tokens=7,
            attributes=json.dumps({"text": "butterfly valve"}),
        ),
        span(
            "c",
            "c-root",
            now - timedelta(days=8),
            seconds=3,
            agent_version="v2",
            conversation_id="other",
        ),
        span("d", "orphan", now - timedelta(minutes=10), parent="missing", status="ERROR"),
        span("e", "e-root", now - timedelta(minutes=20), input_tokens=2, output_tokens=4),
    ]
    write_spans(tmp_path, rows)
    for name in ("answered_question", "grounded", "actionable", "cites_sources"):
        grader(tmp_path, name)
    grader(tmp_path, "answered_question", 2)
    write_grades(
        tmp_path,
        "answered_question",
        [
            ("a", 1, "no", now),
            ("a", 2, "yes", now),
            ("b", 1, "yes", now),
            ("c", 1, "yes", now),
            ("e", 1, "yes", now),
            ("e", 2, "no", now),
        ],
    )
    write_grades(tmp_path, "grounded", [(trace, 1, "no", now) for trace in ("a", "b", "c", "e")])
    for name in ("actionable", "cites_sources"):
        write_grades(tmp_path, name, [("a", 1, "yes", now)])
    return tmp_path, now, rows


def test_all_eight_canonical_queries_and_grade_partition_pruning(canonical_store):
    directory, now, rows = canonical_store
    queries = yaml.safe_load((Path(__file__).parents[1] / "bench/queries.yaml").read_text())
    assert set(queries) == {f"q{index}" for index in range(1, 9)}
    results = {}
    parameters = {
        "q1": {"trace_id": "a"},
        "q3": {"day_start": now - timedelta(days=1), "day_end": now + timedelta(days=1)},
        "q8": {"conversation_id": "conversation"},
    }
    for name, query in queries.items():
        results[name] = dictionaries(execute_sql(directory, query["sql"], parameters.get(name, {})))
    assert [row["span_id"] for row in results["q1"]] == ["a-root", "a-tool", "a-chat"]
    assert [(row["trace_id"], row["span_count"], row["error_count"]) for row in results["q2"]] == [
        ("a", 3, 1)
    ]
    assert [row["span_id"] for row in results["q3"]] == ["a-tool"]
    assert [(row["trace_id"], row["span_count"]) for row in results["q4"]] == [("a", 3), ("b", 1)]
    assert [
        (row["trace_id"], row["answered_question"], row["grounded"]) for row in results["q5"]
    ] == [("a", "yes", "no"), ("b", "yes", "no")]
    assert [row["trace_id"] for row in results["q6"]] == ["a"]
    expected_days = {}
    for _trace, start, duration, tokens, version in [
        ("a", rows[0]["start_time"], 4000, 35, "v1"),
        ("b", rows[3]["start_time"], 2000, 12, "v1"),
        ("c", rows[4]["start_time"], 3000, 0, "v2"),
        ("d", rows[5]["start_time"], 1000, 0, "v1"),
        ("e", rows[6]["start_time"], 1000, 6, "v1"),
    ]:
        key = start.replace(hour=0, minute=0, second=0, microsecond=0), version
        group = expected_days.setdefault(key, {"durations": [], "tokens": 0})
        group["durations"].append(duration)
        group["tokens"] += tokens
    for row in results["q7"]:
        group = expected_days.pop((row["day"], row["agent_version"]))
        durations = sorted(group["durations"])
        position = (len(durations) - 1) * 0.95
        below = int(position)
        above = min(below + 1, len(durations) - 1)
        expected = durations[below] + (durations[above] - durations[below]) * (position - below)
        assert row["p95_duration_ms"] == pytest.approx(expected)
        assert row["total_tokens"] == group["tokens"]
    assert expected_days == {}
    assert [row["trace_id"] for row in results["q8"]] == ["b", "a", "e", "d"]
    with connect(directory) as connection:
        profile_path = directory / "q5-profile.json"
        connection.execute("PRAGMA enable_profiling = 'json'")
        connection.execute("SET profiling_output = ?", [str(profile_path)])
        connection.execute(queries["q5"]["sql"]).fetchall()
        profile = json.loads(profile_path.read_text())
    operators = list(profile["operator"])
    files = set()
    while operators:
        operator = operators.pop()
        operators.extend(operator.get("children", []))
        if operator.get("extra_info", {}).get("Function") == "PARQUET_SCAN":
            files.add(Path(operator["extra_info"]["Filename(s)"]).resolve())
            assert operator["row_groups_scanned"] == 1
    assert files == {
        (directory / "spans/0-1.parquet").resolve(),
        (directory / "grades/grader=answered_question/grades.parquet").resolve(),
        (directory / "grades/grader=grounded/grades.parquet").resolve(),
    }


def test_cli_every_flag_json_profile_sql_and_trace(canonical_store, monkeypatch):
    directory, now, _ = canonical_store
    monkeypatch.setenv("HOPPER_DATA_DIR", str(directory))
    result = runner.invoke(
        app,
        [
            "find",
            "--from",
            (now - timedelta(hours=1)).isoformat(),
            "--to",
            now.isoformat(),
            "--service",
            "rfi-agent",
            "--status",
            "ERROR",
            "--tool",
            "read_drawing",
            "--text",
            "BUTTERFLY VALVE",
            "--grade",
            "answered_question=yes",
            "--grade",
            "grounded=no",
            "--task",
            "task",
            "--conversation",
            "conversation",
            "--limit",
            "1",
            "--json",
            "--profile",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert [(row["trace_id"], row["span_count"]) for row in rows] == [("a", 3)]
    assert rows[0]["actionable"] == "yes"
    assert "wall time:" in result.stderr and "row groups scanned:" in result.stderr
    assert runner.invoke(app, ["find", "--since", "1h", "--status", "ERROR"]).exit_code == 0
    sql = runner.invoke(app, ["sql", "SELECT count(*) AS count FROM spans", "--profile"])
    assert sql.exit_code == 0
    assert sql.stdout == "count\n7\n1 rows\n"
    assert "bytes read:" in sql.stderr
    trace = runner.invoke(app, ["trace", "a", "--full"])
    assert trace.exit_code == 0
    assert "a-root a-root" in trace.stdout and "a-tool a-tool" in trace.stdout
    missing = runner.invoke(app, ["trace", "missing"])
    assert missing.exit_code == 1 and "Trace not found: missing" in missing.stderr


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["find", "--limit", "0"], "--limit must be positive"),
        (["find", "--since", "bad"], "--since must be"),
        (["find", "--since", "1h", "--from", "2026-11-01"], "cannot be combined"),
        (["find", "--from", "bad"], "ISO timestamp"),
        (["find", "--from", "2026-11-02", "--to", "2026-11-01"], "earlier than"),
        (["find", "--grade", "grounded=maybe"], "--grade must be"),
        (["sql", "SELECT missing FROM spans"], "Error:"),
    ],
)
def test_cli_invalid_inputs(tmp_path, monkeypatch, arguments, message):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, arguments)
    assert result.exit_code == 1
    assert message in result.stderr


def test_cli_empty_store_and_message_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    empty = runner.invoke(app, ["find", "--json", "--profile"])
    assert empty.exit_code == 0
    assert json.loads(empty.stdout) == []
    assert "row groups scanned: 0/0" in empty.stderr
    message = "m" * 201
    write_spans(
        tmp_path,
        [
            span(
                "message",
                "root",
                datetime.now(UTC),
                attributes=json.dumps(
                    {
                        "gen_ai.input.messages": json.dumps([message, "n" * 210]),
                        "gen_ai.output.messages": "o" * 220,
                    }
                ),
            )
        ],
    )
    preview = runner.invoke(app, ["trace", "message"])
    full = runner.invoke(app, ["trace", "message", "--full"])
    assert preview.exit_code == full.exit_code == 0
    assert message[:200] + "..." in preview.stdout
    assert message not in preview.stdout
    assert message in full.stdout
    assert "n" * 210 in full.stdout and "o" * 220 in full.stdout
