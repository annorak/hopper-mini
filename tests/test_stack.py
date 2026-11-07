import asyncio
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from hopper import stack, wal
from hopper.config import Settings


def cli(settings, *arguments):
    return subprocess.run(
        [sys.executable, "-c", "from hopper.cli import app; app()", *arguments],
        env={
            **os.environ,
            "HOPPER_DATA_DIR": str(settings.data_dir),
            "HOPPER_ENDPOINT": settings.endpoint,
        },
        capture_output=True,
        text=True,
        timeout=20,
    )


def command(settings, *arguments):
    result = cli(settings, *arguments)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.025)
    pytest.fail("Process did not reach the expected state within 10 seconds")


def send(settings, name):
    request = ExportTraceServiceRequest()
    span = request.resource_spans.add().scope_spans.add().spans.add()
    span.trace_id, span.span_id, span.name = b"t" * 16, b"s" * 8, name
    body = request.SerializeToString()
    response = requests.post(
        settings.endpoint + "/v1/traces",
        data=body,
        headers={"Content-Type": "application/x-protobuf"},
        timeout=5,
    )
    assert response.status_code == 200
    return body


def recovered(settings):
    with wal.Wal.open(settings.data_dir):
        with wal.read_records(settings.data_dir) as records:
            return [record.upload.body for record in records]


@pytest.fixture
def settings():
    with tempfile.TemporaryDirectory(prefix="hopper-stack-", dir="/tmp") as directory:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            endpoint = f"http://127.0.0.1:{listener.getsockname()[1]}"
        config = Settings(Path(directory), endpoint=endpoint)
        try:
            yield config
        finally:
            stack.down(config)


def test_cli_help_and_fsync_choices(settings):
    for arguments in [("ingest", "serve", "--help"), ("stack", "--help")]:
        assert cli(settings, *arguments).returncode == 0
    result = cli(settings, "ingest", "serve", "--fsync", "invalid")
    assert result.returncode == 2
    assert "per-request" in result.stderr


def test_concurrent_up_and_down_are_idempotent(settings):
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=3) as pool:
            starts = list(pool.map(lambda _: command(settings, "stack", "up"), range(3)))
            assert {result["phase"] for result in starts} == {"running"}
            assert len({result["receiver_pid"] for result in starts}) == 1
            assert len({result["supervisor_pid"] for result in starts}) == 1
            assert command(settings, "stack", "status")["phase"] == "running"
            stops = list(pool.map(lambda _: command(settings, "stack", "down"), range(3)))
            assert {result["phase"] for result in stops} == {"stopped"}
        assert command(settings, "stack", "down")["phase"] == "stopped"


def test_sigkill_receiver_restarts_without_acknowledged_loss(settings):
    first = command(settings, "stack", "up")
    acknowledged = [send(settings, f"before kill {index}") for index in range(10)]
    os.kill(first["receiver_pid"], signal.SIGKILL)

    def restarted():
        current = stack.status(settings)
        return (
            current
            if current["phase"] == "running" and current["receiver_pid"] != first["receiver_pid"]
            else None
        )

    second = wait_for(restarted)
    assert second["supervisor_pid"] == first["supervisor_pid"]
    acknowledged.append(send(settings, "after kill"))
    command(settings, "stack", "down")
    assert recovered(settings) == acknowledged
    print(
        json.dumps(
            {
                "sigkill_receiver": first["receiver_pid"],
                "restarted_receiver": second["receiver_pid"],
                "acknowledged_bodies_recovered": len(acknowledged),
            }
        )
    )


def test_sigkill_supervisor_drains_receiver_and_releases_lock(settings):
    running = command(settings, "stack", "up")
    acknowledged = send(settings, "before parent death")
    os.kill(running["supervisor_pid"], signal.SIGKILL)

    def receiver_exited():
        try:
            os.kill(running["receiver_pid"], 0)
            return False
        except ProcessLookupError:
            return True

    wait_for(receiver_exited)
    assert stack.status(settings)["phase"] == "stopped"
    assert recovered(settings) == [acknowledged]
    replacement = command(settings, "stack", "up")
    assert replacement["supervisor_pid"] != running["supervisor_pid"]
    assert replacement["phase"] == "running"


def test_receiver_retries_startup_failures_after_a_running_generation(settings):
    first = command(settings, "stack", "up")
    acknowledged = send(settings, "before startup failures")
    os.kill(first["receiver_pid"], signal.SIGKILL)

    def acquire_wal():
        try:
            return wal.Wal.open(settings.data_dir)
        except RuntimeError:
            return None

    with wait_for(acquire_wal):
        failed = set()
        latest_pid = None

        def failed_generations():
            nonlocal latest_pid
            state = stack.status(settings)
            if state["receiver_pid"] is not None:
                latest_pid = state["receiver_pid"]
            elif latest_pid != first["receiver_pid"] and state["error"] is not None:
                assert "already owns" in state["error"]
                failed.add(latest_pid)
            return len(failed) >= 2

        wait_for(failed_generations)
        state = stack.status(settings)
        assert state["phase"] == "restarting"
        assert state["supervisor_pid"] == first["supervisor_pid"]
    wait_for(lambda: stack.status(settings)["phase"] == "running")
    command(settings, "stack", "down")
    assert recovered(settings) == [acknowledged]


def test_up_waits_for_old_receiver_to_drain_after_supervisor_death(settings):
    first = command(settings, "stack", "up")
    acknowledged = [send(settings, "before immediate replacement")]
    host, port = stack.listen_address(settings.endpoint)
    clients = []
    try:
        for _ in range(8):
            client = socket.create_connection((host, port), timeout=1)
            client.sendall(
                b"POST /v1/traces HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Type: application/x-protobuf\r\nContent-Length: 100\r\n\r\nx"
            )
            clients.append(client)

        def all_slots_occupied():
            response = requests.post(
                settings.endpoint + "/v1/traces",
                data=b"",
                headers={"Content-Type": "application/x-protobuf"},
                timeout=2,
            )
            if response.status_code == 200:
                acknowledged.append(b"")
            return response.status_code == 429

        wait_for(all_slots_occupied)
        os.kill(first["supervisor_pid"], signal.SIGKILL)
        with ThreadPoolExecutor(max_workers=1) as pool:
            replacement = pool.submit(command, settings, "stack", "up")
            wait_for(lambda: "already owns" in (stack.status(settings)["error"] or ""))
            assert not replacement.done()
            for client in clients:
                client.close()
            second = replacement.result(timeout=20)
        assert second["phase"] == "running"
        assert second["receiver_pid"] != first["receiver_pid"]
        command(settings, "stack", "down")
        assert recovered(settings) == acknowledged
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("probe_collision", [False, True])
def test_up_rechecks_state_after_candidate_loses_lock(tmp_path, monkeypatch, probe_collision):
    settings = Settings(tmp_path)
    stopped = {"phase": "stopped", "error": None}
    running = {"phase": "running", "error": None, "receiver_pid": 99}
    states = [stopped, stopped, running]
    if probe_collision:
        states = [stopped, stopped, stopped, running]
    responses = iter(states)
    candidates = []

    def launch(*args, **kwargs):
        candidate = SimpleNamespace(poll=lambda: 0 if len(candidates) == 1 else None)
        candidates.append(candidate)
        return candidate

    monkeypatch.setattr(stack, "status", lambda settings: next(responses))
    monkeypatch.setattr(stack.subprocess, "Popen", launch)
    monkeypatch.setattr(stack.time, "sleep", lambda seconds: None)
    assert stack.up(settings) == running
    assert len(candidates) == (2 if probe_collision else 1)


def test_occupied_port_fails_initial_start_and_cleans_supervisor(settings):
    port = int(settings.endpoint.rsplit(":", 1)[1])
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", port))
        occupied.listen()
        result = cli(settings, "stack", "up")
        assert result.returncode == 1
        assert stack.status(settings)["phase"] == "stopped"
        assert not (settings.data_dir / ".stack" / "control.sock").exists()
        assert (
            "address already in use"
            in (settings.data_dir / ".stack" / "supervisor.log").read_text().lower()
        )
    assert command(settings, "stack", "up")["phase"] == "running"


def test_occupied_wal_fails_initial_start_and_cleans_supervisor(settings):
    with wal.Wal.open(settings.data_dir):
        result = cli(settings, "stack", "up")
        assert result.returncode == 1
        assert stack.status(settings)["phase"] == "stopped"
        assert "already owns" in (settings.data_dir / ".stack" / "supervisor.log").read_text()
    assert command(settings, "stack", "up")["phase"] == "running"


def test_status_distinguishes_starting_lock_from_stopped(settings):
    directory = settings.data_dir / ".stack"
    directory.mkdir()
    with (directory / "owner.lock").open("a+b") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert command(settings, "stack", "status")["phase"] == "starting"
    assert command(settings, "stack", "status")["phase"] == "stopped"


def test_readiness_accepts_fragmented_stream_message():
    async def scenario():
        parent, child = socket.socketpair()
        parent.setblocking(False)
        child.setblocking(False)
        loop = asyncio.get_running_loop()
        try:
            ready = asyncio.create_task(stack._read_ready(parent))
            await loop.sock_sendall(child, b"re")
            await asyncio.sleep(0)
            assert not ready.done()
            await loop.sock_sendall(child, b"ady\n")
            assert await asyncio.wait_for(ready, 1) == b"ready\n"
        finally:
            parent.close()
            child.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("flags", [[], ["--fsync", "per-request"], ["--full-fsync"]])
def test_foreground_cli_stops_and_recovers(settings, flags):
    if flags == ["--full-fsync"] and not hasattr(fcntl, "F_FULLFSYNC"):
        pytest.skip("F_FULLFSYNC is available on macOS")
    with (settings.data_dir / "foreground.log").open("wb") as output:
        child = subprocess.Popen(
            [sys.executable, "-c", "from hopper.cli import app; app()", "ingest", "serve", *flags],
            env={
                **os.environ,
                "HOPPER_DATA_DIR": str(settings.data_dir),
                "HOPPER_ENDPOINT": settings.endpoint,
            },
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:

            def healthy():
                try:
                    return (
                        requests.get(settings.endpoint + "/healthz", timeout=0.2).status_code == 200
                    )
                except requests.ConnectionError:
                    return False

            wait_for(healthy)
            acknowledged = send(settings, "foreground")
        finally:
            with suppress(ProcessLookupError):
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        assert child.returncode == 0, (settings.data_dir / "foreground.log").read_text()
    assert recovered(settings) == [acknowledged]


def test_sigkill_writer_restarts_without_restarting_receiver(settings):
    first = command(settings, "stack", "up")
    send(settings, "writer restart")
    os.kill(first["writer_pid"], signal.SIGKILL)

    def restarted():
        current = stack.status(settings)
        return (
            current
            if current["phase"] == "running" and current["writer_pid"] != first["writer_pid"]
            else None
        )

    second = wait_for(restarted)
    assert second["receiver_pid"] == first["receiver_pid"]
    assert second["supervisor_pid"] == first["supervisor_pid"]
    command(settings, "stack", "down")
    result = cli(settings, "verify")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["actual_spans"] == 1


def test_up_waits_for_old_writer_to_release_ownership(settings):
    from hopper.writer import Writer

    with wal.Wal.open(settings.data_dir):
        pass
    writer = Writer(settings.data_dir)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            startup = pool.submit(command, settings, "stack", "up")
            wait_for(lambda: "writer already owns" in (stack.status(settings)["error"] or ""))
            assert not startup.done()
            writer.close()
            writer = None
            assert startup.result(timeout=20)["phase"] == "running"
    finally:
        if writer is not None:
            writer.close()
