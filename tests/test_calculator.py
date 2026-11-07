import os
import socket
import subprocess
import sys
from unittest.mock import Mock

import pytest

from hopper import calculator


@pytest.fixture
def supported_platform():
    if sys.platform != "darwin":
        pytest.skip("Real sandbox checks require macOS")


@pytest.mark.usefixtures("supported_platform")
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("area = 12 * 8\nprint(area / 3, round(sqrt(area), 2))", "32.0 9.8"),
        ("2 ** 10 + 7 // 2 - 7 % 2", "1026"),
        ("print(-abs(-3), ceil(2.1), floor(2.9), min(3, 2), max(3, 2))", "-3 3 2 2 3"),
        ("round(sin(pi / 2) + log(e), 6)", "2.0"),
        ("x = 12\nx = x + 4\nprint(x)", "16"),
    ],
)
def test_calculate(code, expected):
    assert calculator.calculate(code) == expected


@pytest.mark.usefixtures("supported_platform")
@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from math import sqrt",
        "open('/tmp/calculator-escape', 'w')",
        "__import__('socket').socket()",
        "(1).__class__",
        "[1][0]",
        "while True: pass",
        "for x in [1]: print(x)",
        "def f(): return 1",
        "[x for x in [1]]",
        "print('hello')",
        "print(True)",
        "lambda: 1",
        "sqrt(x=4)",
        "2 ** 101",
        "10 ** 100 * 10 ** 100 * 10 ** 100 * 10 ** 100",
        "1e309",
        "1 / 0",
        "sqrt(-1)",
        "x = y = 1",
        "print = 3",
        "unknown",
        "abs()",
        "1" + " + 1" * 40,
        "print(1)\n" * 130,
        "n = 10 ** 100\n" + "print(n,n,n,n,n,n,n,n)\n" * 6,
    ],
)
def test_rejects_unsupported_or_unbounded_code(code):
    with pytest.raises(calculator.CalculatorInvalid):
        calculator.calculate(code)


def test_source_limit_before_launch(monkeypatch):
    launch = Mock()
    monkeypatch.setattr(calculator.subprocess, "Popen", launch)
    with pytest.raises(calculator.CalculatorInvalid, match="8192"):
        calculator.calculate("1" * 8193)
    launch.assert_not_called()


def test_fails_closed_on_unsupported_platform(monkeypatch):
    monkeypatch.setattr(calculator.sys, "platform", "linux")
    with pytest.raises(calculator.CalculatorSandboxError, match="requires macOS"):
        calculator.calculate("1 + 1")


def test_fails_closed_when_sandbox_missing(monkeypatch):
    monkeypatch.setattr(calculator.sys, "platform", "darwin")
    monkeypatch.setattr(calculator, "_SANDBOX", "/missing-sandbox-executable")
    with pytest.raises(calculator.CalculatorSandboxError):
        calculator.calculate("1 + 1")


def test_worker_start_failure_is_sandbox_error(monkeypatch):
    monkeypatch.setattr(calculator.sys, "platform", "darwin")
    monkeypatch.setattr(calculator.os, "access", lambda *_: True)
    monkeypatch.setattr(calculator.subprocess, "Popen", Mock(side_effect=OSError("denied")))
    with pytest.raises(calculator.CalculatorSandboxError, match="Could not start"):
        calculator.calculate("1")


def test_invalid_worker_response_is_sandbox_error(monkeypatch):
    monkeypatch.setattr(calculator.sys, "platform", "darwin")
    monkeypatch.setattr(calculator.os, "access", lambda *_: True)
    worker = Mock(returncode=0)
    worker.communicate.return_value = ("not json", "")
    monkeypatch.setattr(calculator.subprocess, "Popen", Mock(return_value=worker))
    with pytest.raises(calculator.CalculatorSandboxError, match="invalid response"):
        calculator.calculate("1")


@pytest.mark.usefixtures("supported_platform")
def test_sandbox_denies_file_write_and_network(tmp_path):
    destination = tmp_path / "forbidden"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        probe = (
            "import errno, socket\n"
            f"try:\n open({str(destination)!r}, 'w')\n"
            "except OSError as error:\n assert error.errno == errno.EPERM\n"
            "else:\n raise AssertionError('filesystem write was allowed')\n"
            f"try:\n socket.create_connection(('127.0.0.1', {port}), timeout=1)\n"
            "except OSError as error:\n assert error.errno == errno.EPERM\n"
            "else:\n raise AssertionError('network was allowed')\n"
            "print('denied')\n"
        )
        result = subprocess.run(
            [
                calculator._SANDBOX,
                "-p",
                calculator._PROFILE,
                sys.executable,
                "-I",
                "-S",
                "-c",
                probe,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            env={"LC_ALL": "C"},
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "denied\n"
    assert not destination.exists()


@pytest.mark.usefixtures("supported_platform")
def test_timeout_kills_and_reaps_real_worker(monkeypatch):
    original_popen = subprocess.Popen
    workers = []

    def start_slow_worker(command, **kwargs):
        process = original_popen(
            [*command[:6], "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
        workers.append(process)
        return process

    monkeypatch.setattr(calculator.subprocess, "Popen", start_slow_worker)
    with pytest.raises(calculator.CalculatorTimeout):
        calculator.calculate("1")
    assert len(workers) == 1
    assert workers[0].returncode == -9
    with pytest.raises(ChildProcessError):
        os.waitpid(workers[0].pid, os.WNOHANG)


@pytest.mark.usefixtures("supported_platform")
def test_clean_environment_and_failed_worker(monkeypatch):
    original_popen = subprocess.Popen
    workers = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-child")

    def start_failing_worker(command, **kwargs):
        assert kwargs["close_fds"] is True
        probe = "import os, sys; assert 'ANTHROPIC_API_KEY' not in os.environ; sys.exit(7)"
        process = original_popen([*command[:6], "-c", probe], **kwargs)
        workers.append(process)
        return process

    monkeypatch.setattr(calculator.subprocess, "Popen", start_failing_worker)
    with pytest.raises(calculator.CalculatorSandboxError, match="status 7"):
        calculator.calculate("1")
    with pytest.raises(ChildProcessError):
        os.waitpid(workers[0].pid, os.WNOHANG)


@pytest.mark.usefixtures("supported_platform")
def test_cancellation_kills_and_reaps_real_worker(monkeypatch):
    original_popen = subprocess.Popen
    workers = []

    def start_interrupted_worker(command, **kwargs):
        process = original_popen(
            [*command[:6], "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
        communicate = process.communicate

        def interrupt_once(*args, **kwargs):
            process.communicate = communicate
            raise KeyboardInterrupt

        process.communicate = interrupt_once
        workers.append(process)
        return process

    monkeypatch.setattr(calculator.subprocess, "Popen", start_interrupted_worker)
    with pytest.raises(KeyboardInterrupt):
        calculator.calculate("1")
    assert workers[0].returncode == -9
    with pytest.raises(ChildProcessError):
        os.waitpid(workers[0].pid, os.WNOHANG)
