import base64
import gzip
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from hopper import wal
from hopper.cli import app
from hopper.wal import Upload, Wal

runner = CliRunner()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    return tmp_path


def test_wal_help():
    result = runner.invoke(app, ["wal", "--help"])
    assert result.exit_code == 0
    assert all(command in result.output for command in ("stat", "tail", "prune"))
    for command in ("stat", "tail", "prune"):
        assert runner.invoke(app, ["wal", command, "--help"]).exit_code == 0


def test_stat_on_empty_data_directory(data_dir):
    result = runner.invoke(app, ["wal", "stat"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "segment_count": 0,
        "bytes": 0,
        "physical_end": 0,
        "durable_end": 0,
        "writer_position": 0,
        "backlog": 0,
        "retained_start": 0,
    }
    assert not (data_dir / "wal").exists()


def test_tail_preserves_headers_offsets_and_opaque_gzip_body(data_dir):
    received_at = datetime(2025, 2, 3, 4, 5, 6, 123456, timezone(timedelta(hours=2)))
    body = gzip.compress(b"\x00opaque\xff upload", mtime=0)
    with Wal.open(data_dir) as log:
        first = log.append(Upload(received_at, "application/x-protobuf", b"first", False)).result()
        last = log.append(Upload(received_at, "application/x-protobuf", body, True)).result()
        stat = runner.invoke(app, ["wal", "stat"])
        assert stat.exit_code == 0, stat.output
        assert json.loads(stat.output)["durable_end"] == last.end
    result = runner.invoke(app, ["wal", "tail", "--limit", "1"])
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record == {
        "start": first.end,
        "end": last.end,
        "received_at": "2025-02-03T02:05:06.123456Z",
        "content_type": "application/x-protobuf",
        "is_gzip": True,
        "body_base64": base64.b64encode(body).decode("ascii"),
    }
    assert base64.b64decode(record["body_base64"]) == body


def test_tail_limits(data_dir):
    with Wal.open(data_dir) as log:
        for index in range(12):
            log.append(
                Upload(datetime(2025, 1, 1, tzinfo=UTC), "text/plain", bytes([index]), False)
            )
    result = runner.invoke(app, ["wal", "tail"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.output.splitlines()]
    assert [base64.b64decode(row["body_base64"]) for row in rows] == [
        bytes([index]) for index in range(2, 12)
    ]
    empty = runner.invoke(app, ["wal", "tail", "--limit", "0"])
    assert empty.exit_code == 0
    assert empty.output == ""
    assert runner.invoke(app, ["wal", "tail", "--limit", "-1"]).exit_code == 2


@pytest.mark.parametrize("retention", ["0", "1", "7", "30"])
def test_prune_accepts_retention_days(retention):
    result = runner.invoke(app, ["wal", "prune", "--retention", retention])
    assert result.exit_code == 0, result.output
    assert result.output == "Pruned 0 segment(s).\n"


@pytest.mark.parametrize("retention", ["-1", "7d", "1.5", "nonsense", "0x"])
def test_prune_rejects_invalid_retention(retention):
    result = runner.invoke(app, ["wal", "prune", "--retention", retention])
    assert result.exit_code == 2


def test_prune_rejects_overflow_retention():
    result = runner.invoke(app, ["wal", "prune", "--retention", "999999999999999999999"])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_prune_removes_covered_closed_segments(data_dir, monkeypatch):
    monkeypatch.setattr(wal, "SEGMENT_BYTES", 512)
    with Wal.open(data_dir) as log:
        for _ in range(8):
            end = (
                log.append(
                    Upload(datetime(2025, 1, 1, tzinfo=UTC), "text/plain", b"x" * 100, False)
                )
                .result()
                .end
            )
    segments = sorted((data_dir / "wal").glob("*.seg"))
    assert len(segments) > 1
    (data_dir / "spans").mkdir()
    (data_dir / "spans" / f"0-{end}.parquet").write_bytes(b"committed")
    kept = runner.invoke(app, ["wal", "prune"])
    assert kept.exit_code == 0, kept.output
    assert sorted((data_dir / "wal").glob("*.seg")) == segments
    removed = runner.invoke(app, ["wal", "prune", "--retention", "0"])
    assert removed.exit_code == 0, removed.output
    assert sorted((data_dir / "wal").glob("*.seg")) == segments[-1:]
    assert f"Pruned {len(segments) - 1} segment(s)." in removed.output


@pytest.mark.parametrize("command", ["stat", "tail", "prune"])
def test_corrupt_metadata_reports_error(command, data_dir):
    with Wal.open(data_dir):
        pass
    (data_dir / "wal" / "durable").write_bytes(b"broken")
    result = runner.invoke(app, ["wal", command])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output
