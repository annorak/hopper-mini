import asyncio
import fcntl
import importlib
import json
import logging
import math
import os
import signal
import socket
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from time import monotonic
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from hopper.config import Settings, money
from hopper.corpus import load_corpus
from hopper.graders import Catalog, Grader, JevGrader
from hopper.graders.code import CodeResult
from hopper.graders.jev import JevClient, hash_state
from hopper.graders.state import rfi_core
from hopper.ledger import Ledger, SpendCapExceeded
from hopper.query import GRADE_COLUMNS, connect

BATCH_SIZE = 1_000
POLL_SECONDS = 5
MAX_FAILURES = 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceWork:
    trace_id: str
    revision: int
    seed_revision: int | None
    replay_of: str | None
    pending: tuple[Grader, ...]
    spans: list[dict]

    def key(self, grader: Grader) -> tuple:
        return (self.trace_id, grader.name, grader.version, self.revision, self.seed_revision)


@dataclass
class WorkerStats:
    questions: int = 0
    grades: int = 0
    replayed: int = 0
    failures: int = 0
    seconds: float = 0
    cost_usd: Decimal = Decimal("0")
    held_usd: Decimal = Decimal("0")
    lag_p50_seconds: float | None = None
    lag_p99_seconds: float | None = None
    unsure_rate: dict[str, float] = field(default_factory=dict)
    coverage: list[dict] = field(default_factory=list)
    budget_blocked: str | None = None

    def render(self) -> str:
        return json.dumps(
            {
                "questions": self.questions,
                "questions_per_second": self.questions / self.seconds if self.seconds else 0,
                "grades_written": self.grades,
                "replayed_grades": self.replayed,
                "failures": self.failures,
                "seconds": self.seconds,
                "finished_to_graded_p50_seconds": self.lag_p50_seconds,
                "finished_to_graded_p99_seconds": self.lag_p99_seconds,
                "unsure_rate": self.unsure_rate,
                "cost_usd": str(self.cost_usd),
                "held_usd": str(self.held_usd),
                "coverage": self.coverage,
                "budget_blocked": self.budget_blocked,
            }
        )


@dataclass(frozen=True)
class OnDemandResult:
    grader: str
    grader_version: int
    selected: int
    completed: int
    seed_grades: int
    failed_attempts: int
    cost_usd: Decimal
    held_usd: Decimal
    budget_blocked: str | None

    @property
    def remaining(self) -> int:
        return self.selected - self.completed

    def render(self) -> str:
        return json.dumps({**asdict(self), "remaining": self.remaining}, default=str)


def _rows(cursor) -> list[dict]:
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


async def _finish_mutation[T](operation: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))
    was_cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            was_cancelled = True
    result = task.result()
    if was_cancelled:
        raise asyncio.CancelledError
    return result


class Worker:
    def __init__(
        self,
        settings: Settings,
        *,
        rps: float = 5,
        usd_per_hour: Decimal = Decimal("1"),
        on_demand: Grader | None = None,
    ):
        if not math.isfinite(rps) or rps <= 0:
            raise ValueError("rps must be finite and positive")
        self.settings = settings
        self.rps = rps
        self.usd_per_hour = money(usd_per_hour, "usd_per_hour")
        self.ledger = Ledger(settings)
        self.catalog = Catalog(settings.data_dir)
        self.stats = WorkerStats()
        self._lock = None
        self._client = None
        self._versions = None
        self._history_cursor = None
        self._history_checkpoint = None
        self._history_pending = True
        self._history_page_full = False
        self._deferred_history = False
        self._scan_offset = None
        self._attempts = Counter()
        self._code_functions = {}
        self._on_demand = on_demand

    async def __aenter__(self) -> "Worker":
        await asyncio.to_thread(self.ledger.initialize)
        self._lock = (self.settings.data_dir / ".grader.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            self._lock = None
            raise RuntimeError("Another grader worker owns this data directory") from error
        try:
            graders = (
                [self._on_demand]
                if self._on_demand is not None
                else [g for g in self.catalog.list_graders() if g.status == "active"]
            )
            if (
                any(isinstance(g, JevGrader) for g in graders)
                and not self.settings.typesafe_api_key
            ):
                raise ValueError("TYPESAFE_API_KEY is required for selected Jev graders")
            self._corpus = await asyncio.to_thread(load_corpus)
            for path in (self.settings.data_dir / "tmp").glob("grade-*.parquet"):
                path.unlink()
            for folder in (self.settings.data_dir / "grades").glob("grader=*"):
                _sync_directory(folder)
            _sync_directory(self.settings.data_dir / "grades")
            self._started_at = datetime.now(UTC)
            self._started = monotonic()
            self._starting_cost, _ = self._spend()
            return self
        except BaseException:
            self._lock.close()
            self._lock = None
            raise

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._client is not None:
                await self._client.__aexit__(exc_type, exc_value, traceback)
        finally:
            if self._lock is not None:
                self._lock.close()
                self._lock = None

    def _spend(self) -> tuple[Decimal, Decimal]:
        totals = [total for total in self.ledger.summary() if total.stage == "grading"]
        return (
            sum((total.settled_usd for total in totals), Decimal("0")),
            sum((total.held_usd for total in totals), Decimal("0")),
        )

    def _prepare(self) -> tuple[dict[str, TraceWork], dict[tuple, dict], set[str]]:
        definitions = {
            (g.name, g.version): g for g in self.catalog.list_graders() if g.status == "active"
        }
        versions = frozenset(definitions)
        if versions != self._versions:
            self._history_pending = True
            self._history_cursor = None
            self._versions = versions
        if not self._history_pending and self._deferred_history:
            self._history_pending = True
            self._history_cursor = None
            self._deferred_history = False
        with connect(self.settings.data_dir) as connection:
            maximum = connection.execute(
                "SELECT coalesce(max(log_offset), -1) FROM spans"
            ).fetchone()[0]
            if self._scan_offset is None:
                self._scan_offset = maximum
            connection.execute("""
                CREATE TEMP TABLE exhausted (
                    trace_id VARCHAR, grader VARCHAR, grader_version INTEGER,
                    trace_log_offset BIGINT, seed_log_offset BIGINT
                )
            """)
            exhausted = [key for key, count in self._attempts.items() if count >= MAX_FAILURES]
            if exhausted:
                connection.executemany("INSERT INTO exhausted VALUES (?, ?, ?, ?, ?)", exhausted)
            connection.execute("""
                CREATE TEMP VIEW missing AS
                SELECT t.trace_id, a.grader, a.grader_version
                FROM traces t
                JOIN active_graders a ON a.applies_to = '*' OR a.applies_to = t.service
                LEFT JOIN traces seed ON seed.trace_id = t.replay_of
                LEFT JOIN _fresh_grade_versions g ON g.trace_id = t.trace_id
                    AND g.grader = a.grader AND g.grader_version = a.grader_version
                LEFT JOIN exhausted e ON e.trace_id = t.trace_id AND e.grader = a.grader
                    AND e.grader_version = a.grader_version
                    AND e.trace_log_offset = t.last_log_offset
                    AND e.seed_log_offset IS NOT DISTINCT FROM seed.last_log_offset
                WHERE t.finished AND g.trace_id IS NULL AND e.trace_id IS NULL
            """)
            selection = """
                SELECT t.* FROM traces t LEFT JOIN traces seed ON seed.trace_id = t.replay_of
                WHERE EXISTS (SELECT 1 FROM missing m WHERE m.trace_id = t.trace_id)
            """
            order = " ORDER BY t.last_received_at DESC, t.trace_id DESC LIMIT ?"
            recent = _rows(
                connection.execute(
                    selection
                    + """
                AND (t.last_received_at >= now() - INTERVAL 1 HOUR
                     OR seed.last_received_at >= now() - INTERVAL 1 HOUR)
            """
                    + order,
                    [BATCH_SIZE],
                )
            )
            changed = _rows(
                connection.execute(
                    selection
                    + """
                AND (t.last_log_offset > ? OR seed.last_log_offset > ?)
            """
                    + order,
                    [self._scan_offset, self._scan_offset, BATCH_SIZE],
                )
            )
            if not changed:
                self._scan_offset = maximum
            history = []
            self._history_page_full = False
            self._history_checkpoint = self._history_cursor if self._history_pending else None
            if self._history_pending:
                clause, parameters = "", []
                if self._history_cursor is not None:
                    clause = " AND (t.last_received_at, t.trace_id) < (?, ?)"
                    parameters = list(self._history_cursor)
                history = _rows(
                    connection.execute(selection + clause + order, [*parameters, BATCH_SIZE])
                )
                if history:
                    last = history[-1]
                    self._history_cursor = (last["last_received_at"], last["trace_id"])
                self._history_page_full = len(history) == BATCH_SIZE
                if not self._history_page_full:
                    self._history_pending = False
            selected = {row["trace_id"]: row for row in [*recent, *changed, *history]}
            if not selected:
                return {}, {}, set()
            pending = defaultdict(list)
            for row in _rows(
                connection.execute(
                    "SELECT * FROM missing WHERE trace_id IN (SELECT unnest(?))",
                    [list(selected)],
                )
            ):
                identity = (row["grader"], row["grader_version"])
                if identity in definitions:
                    pending[row["trace_id"]].append(definitions[identity])
            seed_ids = {row["replay_of"] for row in selected.values() if row["replay_of"]}
            seeds = {
                row["trace_id"]: row
                for row in _rows(
                    connection.execute(
                        "SELECT * FROM traces WHERE trace_id IN (SELECT unnest(?))",
                        [sorted(seed_ids)],
                    )
                )
            }
            known = {
                (row["trace_id"], row["grader"], row["grader_version"]): row
                for row in _rows(
                    connection.execute(
                        "SELECT * FROM _fresh_grade_versions WHERE trace_id IN (SELECT unnest(?))",
                        [sorted(seed_ids)],
                    )
                )
            }
            deferred = set()
            for identity, row in list(selected.items()):
                if not row["replay_of"]:
                    continue
                seed = seeds.get(row["replay_of"])
                if seed is None or seed["replay_of"] is not None:
                    continue
                if not seed["finished"]:
                    deferred.add(identity)
                    continue
                selected[seed["trace_id"]] = seed
                for grader in pending[identity]:
                    key = (seed["trace_id"], grader.name, grader.version)
                    attempt = (*key, seed["last_log_offset"], None)
                    if (
                        key not in known
                        and grader not in pending[seed["trace_id"]]
                        and self._attempts[attempt] < MAX_FAILURES
                        and grader.applies_to in ("*", seed["service"])
                    ):
                        pending[seed["trace_id"]].append(grader)
            original_ids = [
                identity
                for identity, row in selected.items()
                if row["replay_of"] is None and pending[identity]
            ]
            spans = defaultdict(list)
            for row in _rows(
                connection.execute(
                    "SELECT * FROM spans WHERE trace_id IN (SELECT unnest(?))",
                    [original_ids],
                )
            ):
                spans[row["trace_id"]].append(row)
            work = {}
            for row in sorted(
                selected.values(),
                key=lambda row: (row["last_received_at"], row["trace_id"]),
                reverse=True,
            ):
                identity = row["trace_id"]
                seed = seeds.get(row["replay_of"])
                work[identity] = TraceWork(
                    identity,
                    row["last_log_offset"],
                    seed["last_log_offset"] if seed else None,
                    row["replay_of"],
                    tuple(pending[identity]),
                    spans[identity],
                )
            self._deferred_history |= bool(deferred)
            return work, known, deferred

    def _fail(self, trace: TraceWork, graders: tuple[Grader, ...], error: Exception) -> None:
        for grader in graders:
            self._attempts[trace.key(grader)] += 1
        self.stats.failures += 1
        logger.error(
            "Grading trace %s with %s failed: %s",
            trace.trace_id,
            ", ".join(f"{g.name} v{g.version}" for g in graders),
            error,
            exc_info=(type(error), error, error.__traceback__),
        )

    def _base_row(self, trace: TraceWork, grader: Grader) -> dict:
        return {
            "trace_id": trace.trace_id,
            "grader": grader.name,
            "grader_version": grader.version,
            "kind": grader.kind,
            "source": "on_demand" if self._on_demand is not None else "live",
            "model": grader.model if isinstance(grader, JevGrader) else None,
            "created_at": datetime.now(UTC),
            "trace_log_offset": trace.revision,
            "seed_log_offset": None,
        }

    async def _grade(self, trace: TraceWork) -> list[dict]:
        try:
            state = await asyncio.to_thread(rfi_core, trace.spans, self._corpus)
        except Exception as error:
            self._fail(trace, trace.pending, error)
            return []
        rows = []
        groups = defaultdict(list)
        for grader in trace.pending:
            if isinstance(grader, JevGrader):
                groups[grader.model].append(grader)
                continue
            try:
                if grader.function not in self._code_functions:
                    module, name = grader.function.split(":")
                    self._code_functions[grader.function] = getattr(
                        importlib.import_module(module), name
                    )
                projected = {field: state[field] for field in grader.fields}
                result = await asyncio.to_thread(
                    self._code_functions[grader.function], projected, self._corpus
                )
                if not isinstance(result, CodeResult) or type(result.passed) is not bool:
                    raise TypeError(
                        "Code graders must return CodeResult with a boolean passed value"
                    )
                if not result.passed and not isinstance(result.reason, str):
                    raise TypeError("A failing code grader must explain its result")
                rows.append(
                    {
                        **self._base_row(trace, grader),
                        "label": "yes" if result.passed else "no",
                        "p_yes": float(result.passed),
                        "reason": result.reason if not result.passed else None,
                        "input_hash": hash_state(projected),
                        "cost_usd": 0.0,
                    }
                )
            except Exception as error:
                self._fail(trace, (grader,), error)
        for graders in groups.values():
            if self.stats.budget_blocked is not None:
                break
            try:
                if self._client is None:
                    self._client = JevClient(
                        self.ledger, rps=self.rps, usd_per_hour=self.usd_per_hour
                    )
                    await self._client.__aenter__()
                judgment = await self._client.judge(state, tuple(graders))
                self.stats.questions += len(judgment.answers)
                definitions = {(g.name, g.version): g for g in graders}
                cost = judgment.cost_usd / len(judgment.answers)
                allocated = Decimal("0")
                for index, answer in enumerate(judgment.answers):
                    share = (
                        judgment.cost_usd - allocated
                        if index == len(judgment.answers) - 1
                        else cost
                    )
                    allocated += share
                    grader = definitions[(answer.grader, answer.grader_version)]
                    rows.append(
                        {
                            **self._base_row(trace, grader),
                            "label": answer.label,
                            "p_yes": answer.p_yes,
                            "reason": None,
                            "input_hash": judgment.input_hash,
                            "cost_usd": float(share),
                        }
                    )
            except SpendCapExceeded as error:
                self.stats.budget_blocked = str(error)
            except Exception as error:
                self._fail(trace, tuple(graders), error)
        return rows

    def _publish(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        with connect(self.settings.data_dir) as connection:
            connection.register("proposed", pa.Table.from_pylist(rows))
            if self._on_demand is not None:
                grader = self._on_demand
                connection.execute(
                    "CREATE TEMP TABLE publication_graders AS "
                    "SELECT ? AS grader, ? AS grader_version, ? AS applies_to",
                    [grader.name, grader.version, grader.applies_to],
                )
            else:
                connection.execute(
                    "CREATE TEMP VIEW publication_graders AS SELECT * FROM active_graders"
                )
            current = _rows(
                connection.execute("""
                SELECT p.* FROM proposed p
                JOIN traces t ON t.trace_id = p.trace_id
                JOIN publication_graders a ON a.grader = p.grader
                    AND a.grader_version = p.grader_version
                LEFT JOIN traces seed ON seed.trace_id = t.replay_of
                WHERE t.finished AND (a.applies_to = '*' OR a.applies_to = t.service)
                  AND p.trace_log_offset = t.last_log_offset
                  AND p.seed_log_offset IS NOT DISTINCT FROM seed.last_log_offset
            """)
            )
        grouped = defaultdict(list)
        for row in current:
            grouped[row["grader"]].append(row)
        arrow_types = {
            "VARCHAR": pa.string(),
            "INTEGER": pa.int32(),
            "BIGINT": pa.int64(),
            "DOUBLE": pa.float64(),
            "TIMESTAMPTZ": pa.timestamp("us", tz="UTC"),
        }
        schema = pa.schema([(name, arrow_types[kind]) for name, kind in GRADE_COLUMNS.items()])
        for name, batch in grouped.items():
            folder = self.settings.data_dir / "grades" / f"grader={name}"
            folder.mkdir(exist_ok=True)
            identity = uuid4().hex
            temporary = self.settings.data_dir / "tmp" / f"grade-{identity}.parquet"
            pq.write_table(
                pa.Table.from_pylist(batch, schema=schema), temporary, compression="zstd"
            )
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, folder / f"{identity}.parquet")
            _sync_directory(folder)
        _sync_directory(self.settings.data_dir / "grades")
        return current

    def _on_demand_work(self, trace_ids: list[str]) -> list[TraceWork]:
        grader = self._on_demand
        with connect(self.settings.data_dir) as connection:
            selected = _rows(
                connection.execute(
                    "SELECT * FROM traces WHERE trace_id IN (SELECT unnest(?)) "
                    "AND finished AND replay_of IS NULL AND (? = '*' OR service = ?)",
                    [trace_ids, grader.applies_to, grader.applies_to],
                )
            )
            spans = defaultdict(list)
            for row in _rows(
                connection.execute(
                    "SELECT * FROM spans WHERE trace_id IN (SELECT unnest(?))",
                    [[row["trace_id"] for row in selected]],
                )
            ):
                spans[row["trace_id"]].append(row)
        return [
            TraceWork(
                row["trace_id"],
                row["last_log_offset"],
                None,
                None,
                (grader,),
                spans[row["trace_id"]],
            )
            for row in selected
        ]

    async def _grade_on_demand_page(self, trace_ids: list[str]) -> set[str]:
        completed = set()
        for _ in range(MAX_FAILURES):
            pending = [identity for identity in trace_ids if identity not in completed]
            if not pending or self.stats.budget_blocked:
                break
            work = await asyncio.to_thread(self._on_demand_work, pending)
            rows = await self._grade_originals(work)
            published = await _finish_mutation(partial(self._publish, rows))
            completed.update(row["trace_id"] for row in published)
        self._attempts.clear()
        for identity in set(trace_ids) - completed:
            logger.error(
                "On-demand trace %s has no published grade after at most %s attempts",
                identity,
                MAX_FAILURES,
            )
        return completed

    def _on_demand_replays(self, trace_ids: list[str]) -> list[dict]:
        grader = self._on_demand
        with connect(self.settings.data_dir) as connection:
            return _rows(
                connection.execute(
                    """
                SELECT g.* REPLACE (
                    t.trace_id AS trace_id, 'on_demand' AS source, 0.0::DOUBLE AS cost_usd,
                    now() AS created_at, t.last_log_offset AS trace_log_offset,
                    seed.last_log_offset AS seed_log_offset
                )
                FROM traces t JOIN traces seed ON seed.trace_id = t.replay_of
                JOIN _fresh_grade_versions g ON g.trace_id = seed.trace_id
                    AND g.grader = ? AND g.grader_version = ?
                WHERE t.trace_id IN (SELECT unnest(?)) AND t.finished AND seed.finished
                    AND seed.replay_of IS NULL
                    AND (? = '*' OR (t.service = ? AND seed.service = ?))
                """,
                    [
                        grader.name,
                        grader.version,
                        trace_ids,
                        grader.applies_to,
                        grader.applies_to,
                        grader.applies_to,
                    ],
                )
            )

    async def _copy_on_demand_page(self, trace_ids: list[str]) -> int:
        pending = set(trace_ids)
        for _ in range(MAX_FAILURES):
            if not pending:
                break
            rows = await asyncio.to_thread(self._on_demand_replays, sorted(pending))
            if not rows:
                break
            published = await _finish_mutation(partial(self._publish, rows))
            pending.difference_update(row["trace_id"] for row in published)
        for identity in sorted(pending):
            logger.error(
                "On-demand replay %s has no current grade: its original seed must be finished "
                "and applicable with a fresh %s v%s grade, and both revisions must stay current",
                identity,
                self._on_demand.name,
                self._on_demand.version,
            )
        return len(trace_ids) - len(pending)

    async def run_on_demand(self, *, where: str, limit: int | None) -> OnDemandResult:
        if self._lock is None or self._on_demand is None:
            raise RuntimeError("On-demand grading requires a selected grader and worker context")
        grader = self._on_demand
        completed = 0
        seed_grades = 0
        with connect(self.settings.data_dir) as selection:
            predicate = f"SELECT trace_id FROM traces WHERE ({where})"
            statements = selection.extract_statements(predicate)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise ValueError("--where must be one SQL predicate over traces")
            await _finish_mutation(
                partial(
                    selection.execute,
                    "CREATE TEMP TABLE selected AS SELECT * FROM traces "
                    f"WHERE trace_id IN ({predicate}) AND finished "
                    "AND (? = '*' OR service = ?) "
                    "ORDER BY last_received_at DESC, trace_id DESC LIMIT ?",
                    [grader.applies_to, grader.applies_to, limit],
                )
            )
            selected = selection.execute("SELECT count(*) FROM selected").fetchone()[0]
            cursor = await _finish_mutation(
                partial(
                    selection.execute,
                    """
                SELECT t.trace_id, EXISTS (
                    SELECT 1 FROM selected s WHERE s.trace_id = t.trace_id
                ) AS is_target
                FROM traces t
                WHERE t.trace_id IN (SELECT coalesce(replay_of, trace_id) FROM selected)
                    AND t.replay_of IS NULL AND t.finished
                    AND (? = '*' OR t.service = ?)
                    AND (EXISTS (SELECT 1 FROM selected s WHERE s.trace_id = t.trace_id)
                        OR NOT EXISTS (SELECT 1 FROM _fresh_grade_versions g
                            WHERE g.trace_id = t.trace_id AND g.grader = ?
                                AND g.grader_version = ?))
                ORDER BY t.last_received_at DESC, t.trace_id DESC
                """,
                    [grader.applies_to, grader.applies_to, grader.name, grader.version],
                )
            )
            while page := cursor.fetchmany(BATCH_SIZE):
                published = await self._grade_on_demand_page([identity for identity, _ in page])
                completed += sum(
                    identity in published and is_target for identity, is_target in page
                )
                seed_grades += sum(
                    identity in published and not is_target for identity, is_target in page
                )
                if self.stats.budget_blocked:
                    break
            cursor = selection.execute(
                "SELECT trace_id FROM selected WHERE replay_of IS NOT NULL "
                "ORDER BY last_received_at DESC, trace_id DESC"
            )
            while page := cursor.fetchmany(BATCH_SIZE):
                completed += await self._copy_on_demand_page([row[0] for row in page])
        cost, held = self._spend()
        return OnDemandResult(
            grader.name,
            grader.version,
            selected,
            completed,
            seed_grades,
            self.stats.failures,
            cost - self._starting_cost,
            held,
            self.stats.budget_blocked,
        )

    def _has_coverage(self, grader: Grader) -> bool:
        with connect(self.settings.data_dir) as connection:
            return not connection.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM traces t
                    LEFT JOIN _fresh_grade_versions g ON g.trace_id = t.trace_id
                        AND g.grader = ? AND g.grader_version = ?
                    WHERE t.finished AND (? = '*' OR t.service = ?)
                        AND g.trace_id IS NULL
                )
            """,
                [grader.name, grader.version, grader.applies_to, grader.applies_to],
            ).fetchone()[0]

    def _refresh_metrics(self) -> None:
        self.stats.seconds = monotonic() - self._started
        cost, held = self._spend()
        self.stats.cost_usd = cost - self._starting_cost
        self.stats.held_usd = held
        with connect(self.settings.data_dir) as connection:
            self.stats.unsure_rate = {
                name: rate
                for name, rate in connection.execute(
                    """
                    SELECT grader, avg(CASE WHEN label = 'unsure' THEN 1.0 ELSE 0.0 END)
                    FROM grades WHERE created_at >= ? AND source = 'live' GROUP BY grader
                """,
                    [self._started_at],
                ).fetchall()
            }
            self.stats.coverage = _rows(
                connection.execute("""
                SELECT a.grader, a.grader_version, count(t.trace_id) AS finished,
                    count(g.trace_id) AS graded,
                    count(t.trace_id) - count(g.trace_id) AS missing
                FROM active_graders a
                LEFT JOIN traces t ON t.finished
                    AND (a.applies_to = '*' OR t.service = a.applies_to)
                LEFT JOIN _fresh_grade_versions g ON g.trace_id = t.trace_id
                    AND g.grader = a.grader AND g.grader_version = a.grader_version
                GROUP BY a.grader, a.grader_version ORDER BY a.grader, a.grader_version
            """)
            )
            self.stats.lag_p50_seconds, self.stats.lag_p99_seconds = connection.execute(
                """
                WITH completed AS (
                    SELECT t.trace_id, t.last_received_at, t.has_root, t.service,
                        max(g.created_at) AS graded_at, count(*) AS grades
                    FROM traces t JOIN _fresh_grade_versions g ON g.trace_id = t.trace_id
                    JOIN active_graders a ON a.grader = g.grader
                        AND a.grader_version = g.grader_version
                    WHERE t.finished AND t.replay_of IS NULL
                        AND (a.applies_to = '*' OR a.applies_to = t.service)
                    GROUP BY t.trace_id, t.last_received_at, t.has_root, t.service
                ), lags AS (
                    SELECT epoch(graded_at - last_received_at)
                        - CASE WHEN has_root THEN 10 ELSE 600 END AS seconds
                    FROM completed c WHERE graded_at >= ? AND grades = (
                        SELECT count(*) FROM active_graders a
                        WHERE a.applies_to = '*' OR a.applies_to = c.service
                    )
                )
                SELECT quantile_cont(seconds, 0.5), quantile_cont(seconds, 0.99) FROM lags
            """,
                [self._started_at],
            ).fetchone()

    async def _grade_originals(self, work: list[TraceWork]) -> list[dict]:
        originals = iter(work)
        rows = []

        async def consume() -> None:
            for trace in originals:
                rows.extend(await self._grade(trace))

        async with asyncio.TaskGroup() as tasks:
            for _ in range(8):
                tasks.create_task(consume())
        return rows

    async def run_pass(self) -> WorkerStats:
        if self._lock is None:
            raise RuntimeError("Use Worker as an async context manager")
        if self._on_demand is not None:
            raise RuntimeError("On-demand workers cannot run automatic passes")
        self.stats.budget_blocked = None
        failures = self.stats.failures
        work, known, deferred = await asyncio.to_thread(self._prepare)
        rows = await self._grade_originals(
            [trace for trace in work.values() if trace.replay_of is None and trace.pending]
        )
        for row in rows:
            known[(row["trace_id"], row["grader"], row["grader_version"])] = row
        for trace in work.values():
            if trace.replay_of is None or trace.trace_id in deferred:
                continue
            for grader in trace.pending:
                seed = known.get((trace.replay_of, grader.name, grader.version))
                if seed is None:
                    seed_work = work.get(trace.replay_of)
                    if (
                        seed_work is None
                        or seed_work.replay_of is not None
                        or grader not in seed_work.pending
                    ):
                        self._fail(
                            trace, (grader,), ValueError("Replay seed is missing or inapplicable")
                        )
                    continue
                rows.append(
                    {
                        **seed,
                        "trace_id": trace.trace_id,
                        "source": "replayed",
                        "cost_usd": 0.0,
                        "created_at": datetime.now(UTC),
                        "trace_log_offset": trace.revision,
                        "seed_log_offset": seed["trace_log_offset"],
                    }
                )
        published = await _finish_mutation(lambda: self._publish(rows))
        self.stats.grades += len(published)
        self.stats.replayed += sum(row["source"] == "replayed" for row in published)
        if (
            self.stats.failures != failures
            or self.stats.budget_blocked
            or len(published) != len(rows)
        ):
            self._history_pending = True
            self._history_cursor = self._history_checkpoint
        retired = await _finish_mutation(lambda: self.catalog.retire_superseded(self._has_coverage))
        for grader in retired:
            logger.info("Retired covered grader %s v%s", grader.name, grader.version)
        await asyncio.to_thread(self._refresh_metrics)
        return self.stats


async def run_on_demand(
    settings: Settings,
    *,
    grader: str,
    version: int | None,
    where: str,
    limit: int | None,
) -> OnDemandResult:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if version is not None and version <= 0:
        raise ValueError("--version must be positive")
    if not where.strip():
        raise ValueError("--where must be a SQL predicate over traces")
    await asyncio.to_thread(Ledger(settings).initialize)
    definitions = [
        definition
        for definition in Catalog(settings.data_dir).list_graders()
        if definition.name == grader and (version is None or definition.version == version)
    ]
    if not definitions:
        suffix = f" version {version}" if version is not None else ""
        raise ValueError(f"Unknown grader {grader}{suffix}")
    definition = max(definitions, key=lambda item: item.version)
    async with Worker(settings, on_demand=definition) as worker:
        return await worker.run_on_demand(where=where, limit=limit)


async def run_worker(
    settings: Settings,
    *,
    rps: float = 5,
    usd_per_hour: Decimal = Decimal("1"),
    once: bool = False,
    parent: socket.socket | None = None,
) -> WorkerStats:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    monitor = None
    try:
        async with Worker(settings, rps=rps, usd_per_hour=usd_per_hour) as worker:
            if parent is not None:
                parent.setblocking(False)
                await loop.sock_sendall(parent, b"ready\n")

                async def watch_parent() -> None:
                    try:
                        while await loop.sock_recv(parent, 1):
                            pass
                    finally:
                        stop.set()

                monitor = asyncio.create_task(watch_parent())
            previous = None
            while not stop.is_set():
                grading = asyncio.create_task(worker.run_pass())
                stopping = asyncio.create_task(stop.wait())
                try:
                    await asyncio.wait((grading, stopping), return_when=asyncio.FIRST_COMPLETED)
                    if stopping.done():
                        grading.cancel()
                        with suppress(asyncio.CancelledError):
                            await grading
                        break
                    stats = await grading
                finally:
                    grading.cancel()
                    stopping.cancel()
                    outcomes = await asyncio.gather(grading, stopping, return_exceptions=True)
                    for outcome in outcomes:
                        if isinstance(outcome, Exception):
                            raise outcome
                print(stats.render(), flush=True)
                progress = (stats.grades, stats.failures)
                if once:
                    if stats.budget_blocked or (
                        not worker._history_page_full and progress == previous
                    ):
                        break
                    previous = progress
                    continue
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
            return worker.stats
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    with socket.socket(fileno=int(os.environ["HOPPER_PARENT_FD"])) as parent_socket:
        try:
            asyncio.run(run_worker(Settings.from_env(), parent=parent_socket))
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
            with suppress(OSError):
                parent_socket.settimeout(1)
                kind = "grader-busy" if isinstance(error.__cause__, BlockingIOError) else "error"
                parent_socket.sendall(f"{kind} {error}\n".encode()[:2048])
            raise SystemExit(1) from error
