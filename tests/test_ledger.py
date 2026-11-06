import multiprocessing
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import Barrier, Event
from unittest.mock import Mock

import pytest

from hopper.config import (
    QuestionBudget,
    QuestionUsage,
    Settings,
    TokenBudget,
    TokenUsage,
    token_cost,
)
from hopper.ledger import CallResult, Ledger, SpendCapExceeded, UsageLimitExceeded

MODEL = "claude-haiku-5-5"
BUDGET = TokenBudget(MODEL, 0, 100_000)


@pytest.fixture
def ledger(tmp_path):
    ledger = Ledger(Settings(tmp_path, Decimal("0.05")))
    ledger.initialize()
    return ledger


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (TokenUsage(100_000, 10), "0.010005"),
        (TokenUsage(100_001, 10), "0.0500255"),
        (TokenUsage(0, 0, 100_001), "0.00500005"),
        (TokenUsage(0, 0, 0, 100_001), "0.062500625"),
    ],
)
def test_haiku_tier_includes_all_input_categories(usage, expected):
    assert token_cost(MODEL, usage) == Decimal(expected)


def test_sonnet_prices():
    usage = TokenUsage(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert token_cost("claude-sonnet-5-5", usage) == Decimal("14.6")


def test_exact_cap_then_crossing_is_refused_before_request(ledger, monkeypatch):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(ledger.settings.data_dir))
    monkeypatch.setenv("HOPPER_SPEND_CAP_USD", "0.05")
    ledger = Ledger(Settings.from_env())
    request = Mock(return_value=CallResult("ok", TokenUsage(0, 100_000)))
    assert ledger.call("seed", BUDGET, request) == "ok"
    request.assert_called_once()
    refused = Mock()
    with pytest.raises(SpendCapExceeded):
        Ledger(Settings.from_env()).call("seed", BUDGET, refused)
    refused.assert_not_called()
    assert ledger.summary()[0].settled_usd == Decimal("0.05")


def test_cache_write_price_is_reserved_before_request(ledger):
    request = Mock()
    with pytest.raises(SpendCapExceeded):
        ledger.call("seed", TokenBudget(MODEL, 100_000, 80_000), request)
    request.assert_not_called()


def test_jev_question_cost_and_spend_output(ledger, monkeypatch):
    from typer.testing import CliRunner

    from hopper.cli import app

    result = CallResult("yes", QuestionUsage(7))
    ledger.call("grade", QuestionBudget("jev", 10), Mock(return_value=result))
    assert ledger.summary()[0].settled_usd == Decimal("0.00049")
    monkeypatch.setenv("HOPPER_DATA_DIR", str(ledger.settings.data_dir))
    output = CliRunner().invoke(app, ["spend"])
    assert output.exit_code == 0
    assert "grade\tjev\t1\t0.00049\t0" in output.output


def test_request_failure_retains_reservation(ledger):
    with pytest.raises(ConnectionError):
        ledger.call("seed", BUDGET, Mock(side_effect=ConnectionError("unknown outcome")))
    assert Ledger(ledger.settings).summary()[0].held_usd == Decimal("0.05")
    refused = Mock()
    with pytest.raises(SpendCapExceeded):
        ledger.call("seed", BUDGET, refused)
    refused.assert_not_called()


def test_wrong_usage_retains_reservation(ledger):
    with pytest.raises(ValueError, match="TokenUsage"):
        ledger.call("seed", BUDGET, Mock(return_value=CallResult("bad", QuestionUsage(1))))
    assert ledger.summary()[0].held_usd == Decimal("0.05")


def test_actual_overrun_is_recorded_before_error(ledger):
    request = Mock(return_value=CallResult("too much", TokenUsage(0, 200_000)))
    with pytest.raises(UsageLimitExceeded):
        ledger.call("seed", BUDGET, request)
    assert ledger.summary()[0].settled_usd == Decimal("0.1")
    assert ledger.summary()[0].held_usd == 0


def test_concurrent_calls_settle_in_reverse_order(tmp_path):
    ledger = Ledger(Settings(tmp_path, Decimal("0.1")))
    ledger.initialize()
    ready = Barrier(3)
    release = [Event(), Event()]

    def request(index):
        ready.wait(timeout=5)
        assert release[index].wait(timeout=5)
        return CallResult(index, TokenUsage(0, (2 - index) * 25_000))

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(ledger.call, "seed", BUDGET, lambda index=index: request(index))
            for index in range(2)
        ]
        try:
            ready.wait(timeout=5)
            assert ledger.summary()[0].held_usd == Decimal("0.1")
            refused = Mock()
            with pytest.raises(SpendCapExceeded):
                ledger.call("seed", BUDGET, refused)
            refused.assert_not_called()
            release[1].set()
            assert futures[1].result(timeout=5) == 1
            release[0].set()
            assert futures[0].result(timeout=5) == 0
        finally:
            for event in release:
                event.set()
    total = Ledger(ledger.settings).summary()[0]
    assert total.calls == 2
    assert total.held_usd == 0
    assert total.settled_usd == Decimal("0.0375")


def _request_until_killed(settings, connection):
    def request():
        connection.send("request sent")
        signal.pause()
        raise AssertionError("Expected SIGKILL")

    Ledger(settings).call("seed", BUDGET, request)


def test_sigkill_preserves_reservation(ledger):
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_request_until_killed, args=(ledger.settings, sending))
    process.start()
    sending.close()
    try:
        assert receiving.poll(5)
        assert receiving.recv() == "request sent"
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        receiving.close()
        process.close()
    restarted = Ledger(ledger.settings)
    assert restarted.summary()[0].held_usd == Decimal("0.05")
    refused = Mock()
    with pytest.raises(SpendCapExceeded):
        restarted.call("seed", BUDGET, refused)
    refused.assert_not_called()
