import asyncio
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from hopper.config import Settings
from hopper.edge import listen_address

START_SECONDS = 10
STOP_SECONDS = 5
POLL_SECONDS = 0.05


async def _read_ready(parent: socket.socket) -> bytes:
    message = bytearray()
    loop = asyncio.get_running_loop()
    while len(message) < 4096 and not message.endswith(b"\n"):
        chunk = await loop.sock_recv(parent, 4096 - len(message))
        if not chunk:
            break
        message.extend(chunk)
    return bytes(message)


def _control(data_dir: Path, command: str) -> dict:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(1)
        client.connect(str(data_dir / ".stack" / "control.sock"))
        client.sendall(command.encode() + b"\n")
        with client.makefile("rb") as reader:
            return json.loads(reader.readline(4096))


def _locked(data_dir: Path) -> bool:
    path = data_dir / ".stack" / "owner.lock"
    if not path.exists():
        return False
    with path.open("a+b") as owner:
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def status(settings: Settings) -> dict:
    try:
        return _control(settings.data_dir, "status")
    except (OSError, ValueError):
        return {
            "phase": "starting" if _locked(settings.data_dir) else "stopped",
            "supervisor_pid": None,
            "receiver_pid": None,
            "error": None,
        }


def up(settings: Settings) -> dict:
    listen_address(settings.endpoint)
    directory = settings.data_dir / ".stack"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate = None
    failure = None
    deadline = time.monotonic() + START_SECONDS + STOP_SECONDS
    while time.monotonic() < deadline:
        result = status(settings)
        if candidate is not None and (exit_code := candidate.poll()) is not None:
            result = status(settings)
            if exit_code != 0 and result["phase"] == "stopped":
                raise RuntimeError(
                    failure or f"Receiver failed to start; see {directory / 'supervisor.log'}"
                )
            candidate = None
        if result["phase"] == "running":
            return result
        failure = result["error"] or failure
        if candidate is None and result["phase"] == "stopped":
            env = {
                **os.environ,
                "HOPPER_DATA_DIR": str(settings.data_dir),
                "HOPPER_ENDPOINT": settings.endpoint,
            }
            with (directory / "supervisor.log").open("ab", buffering=0) as output:
                candidate = subprocess.Popen(
                    [sys.executable, "-m", "hopper.stack"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=True,
                )
        time.sleep(POLL_SECONDS)
    if candidate is not None and candidate.poll() is None:
        with suppress(ProcessLookupError):
            candidate.terminate()
        try:
            candidate.wait(timeout=STOP_SECONDS + 1)
        except subprocess.TimeoutExpired:
            candidate.kill()
            candidate.wait(timeout=STOP_SECONDS)
    raise RuntimeError(f"Receiver startup timed out; see {directory / 'supervisor.log'}")


def down(settings: Settings) -> dict:
    deadline = time.monotonic() + START_SECONDS + STOP_SECONDS
    requested = False
    while time.monotonic() < deadline:
        if not _locked(settings.data_dir):
            return status(settings)
        if not requested:
            try:
                _control(settings.data_dir, "down")
                requested = True
            except (OSError, ValueError):
                pass
        time.sleep(POLL_SECONDS)
    raise RuntimeError("Supervisor did not stop within the shutdown deadline")


class Supervisor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.stop = asyncio.Event()
        self.phase = "starting"
        self.child: subprocess.Popen | None = None
        self.error: str | None = None

    def snapshot(self) -> dict:
        return {
            "phase": self.phase,
            "supervisor_pid": os.getpid(),
            "receiver_pid": self.child.pid if self.child is not None else None,
            "error": self.error,
        }

    async def control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            async with asyncio.timeout(1):
                command = await reader.readline()
                if command == b"down\n":
                    self.phase = "stopping"
                    self.stop.set()
                elif command != b"status\n":
                    raise ValueError("Unknown control command")
                writer.write(json.dumps(self.snapshot()).encode() + b"\n")
                await writer.drain()
        except (OSError, ValueError, TimeoutError):
            pass
        finally:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()

    async def stop_child(self) -> None:
        if self.child is None:
            return
        if self.child.poll() is None:
            with suppress(ProcessLookupError):
                self.child.terminate()
        try:
            await asyncio.to_thread(self.child.wait, timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            self.child.kill()
            await asyncio.to_thread(self.child.wait, timeout=STOP_SECONDS)
        self.child = None

    async def run_children(self) -> None:
        loop = asyncio.get_running_loop()
        has_run = False
        failures = 0
        initial_deadline = loop.time() + START_SECONDS + STOP_SECONDS
        while not self.stop.is_set():
            parent, child = socket.socketpair()
            parent.setblocking(False)
            stopped = asyncio.create_task(self.stop.wait())
            ready = None
            can_retry_startup = False
            try:
                env = {
                    **os.environ,
                    "HOPPER_DATA_DIR": str(self.settings.data_dir),
                    "HOPPER_ENDPOINT": self.settings.endpoint,
                    "HOPPER_PARENT_FD": str(child.fileno()),
                }
                self.child = subprocess.Popen(
                    [sys.executable, "-m", "hopper.edge"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    pass_fds=(child.fileno(),),
                )
                child.close()
                ready = asyncio.create_task(_read_ready(parent))
                done, _ = await asyncio.wait(
                    [ready, stopped],
                    timeout=START_SECONDS if has_run else max(0, initial_deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self.stop.is_set():
                    break
                if ready not in done:
                    raise RuntimeError("Receiver timed out before readiness")
                if ready.result() != b"ready\n":
                    kind, _, message = ready.result().partition(b" ")
                    can_retry_startup = kind == b"wal-busy" and loop.time() < initial_deadline
                    raise RuntimeError(
                        message.decode(errors="replace").strip()
                        or "Receiver exited before readiness"
                    )
                self.phase = "running"
                self.error = None
                has_run = True
                started = loop.time()
                while self.child.poll() is None and not self.stop.is_set():
                    await asyncio.wait([stopped], timeout=POLL_SECONDS)
                self.phase = "stopping" if self.stop.is_set() else "restarting"
                if loop.time() - started >= 1:
                    failures = 0
            except (OSError, RuntimeError) as error:
                self.error = str(error)
            finally:
                child.close()
                parent.close()
                stopped.cancel()
                if ready is not None:
                    ready.cancel()
                await asyncio.gather(
                    stopped, *([ready] if ready is not None else []), return_exceptions=True
                )
                await self.stop_child()
            if not has_run and not can_retry_startup:
                raise RuntimeError(self.error or "Receiver stopped before readiness")
            if not self.stop.is_set():
                self.phase = "restarting" if has_run else "starting"
                failures += 1
                delay = min(0.1 * 2 ** min(failures - 1, 4), 1)
                if not has_run:
                    delay = min(delay, max(0, initial_deadline - loop.time()))
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.stop.wait(), delay)


async def supervise(settings: Settings) -> None:
    listen_address(settings.endpoint)
    directory = settings.data_dir / ".stack"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "control.sock"
    with (directory / "owner.lock").open("a+b") as owner:
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        path.unlink(missing_ok=True)
        supervisor = Supervisor(settings)
        server = await asyncio.start_unix_server(supervisor.control, path=str(path), limit=4096)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, supervisor.stop.set)
        try:
            async with server:
                await supervisor.run_children()
        finally:
            supervisor.stop.set()
            await supervisor.stop_child()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(supervise(Settings.from_env()))
