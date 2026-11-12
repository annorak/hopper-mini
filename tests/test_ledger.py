import asyncio
import multiprocessing
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier, Event
from unittest.mock import AsyncMock, Mock

import pytest

from hopper.config import (
    JevBudget,
    JevUsage,
    Settings,
    TokenBudget,
    TokenUsage,
    token_cost,
)
from hopper.ledger import (
    CallResult,
    HourlySpendCapExceeded,
    Ledger,
    SpendCapExceeded,
    UsageLimitExceeded,
)

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


def test_jev_input_token_cost_and_spend_output(ledger, monkeypatch):
    from typer.testing import CliRunner

    from hopper.cli import app

    result = CallResult("yes", JevUsage(7000))
    ledger.call("grade", JevBudget("jev-1.14.0", 10_000), Mock(return_value=result))
    assert ledger.summary()[0].settled_usd == Decimal("0.000294")
    monkeypatch.setenv("HOPPER_DATA_DIR", str(ledger.settings.data_dir))
    output = CliRunner().invoke(app, ["spend"])
    assert output.exit_code == 0
    assert "grade\tjev-1.14.0\t1\t0.000294\t0" in output.output


def test_jev_budget_rejects_before_dispatch(ledger):
    request = Mock()
    with pytest.raises(SpendCapExceeded):
        ledger.call("grade", JevBudget("jev-1.14.0", 2_000_000), request)
    request.assert_not_called()


def test_jev_overrun_is_settled_before_error(ledger):
    request = Mock(return_value=CallResult("yes", JevUsage(20_000)))
    with pytest.raises(UsageLimitExceeded):
        ledger.call("grade", JevBudget("jev-1.14.0", 10_000), request)
    total = ledger.summary()[0]
    assert total.settled_usd == Decimal("0.00084")
    assert total.held_usd == 0


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
        ledger.call("seed", BUDGET, Mock(return_value=CallResult("bad", JevUsage(1))))
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


def test_async_call_reserves_before_dispatch_and_returns_charge(ledger):
    async def request():
        assert ledger.summary()[0].held_usd == Decimal("0.00042")
        return CallResult("yes", JevUsage(7000))

    receipt = asyncio.run(
        ledger.call_async(
            "grade", JevBudget("jev-1.14.0", 10_000), request, usd_per_hour=Decimal("1")
        )
    )
    assert receipt.value == "yes"
    assert receipt.cost_usd == Decimal("0.000294")
    assert ledger.summary()[0].held_usd == 0


def test_async_global_cap_refuses_before_dispatch(ledger):
    request = AsyncMock(return_value=CallResult("yes", JevUsage(1)))
    with pytest.raises(SpendCapExceeded):
        asyncio.run(
            ledger.call_async(
                "grade", JevBudget("jev-1.14.0", 2_000_000), request, usd_per_hour=None
            )
        )
    request.assert_not_awaited()


def test_async_cancellation_retains_reservation(ledger):
    async def run():
        started = asyncio.Event()

        async def request():
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("Expected cancellation")

        task = asyncio.create_task(
            ledger.call_async(
                "grade", JevBudget("jev-1.14.0", 10_000), request, usd_per_hour=Decimal("1")
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert Ledger(ledger.settings).summary()[0].held_usd == Decimal("0.00042")


def test_async_hourly_cap_covers_concurrent_holds_and_restart(ledger):
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        budget = JevBudget("jev-1.14.0", 10_000)
        limit = Decimal("0.00042")

        async def request():
            started.set()
            await release.wait()
            return CallResult("yes", JevUsage(10_000))

        task = asyncio.create_task(ledger.call_async("grade", budget, request, usd_per_hour=limit))
        refused = AsyncMock(return_value=CallResult("yes", JevUsage(1)))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            with pytest.raises(HourlySpendCapExceeded):
                await Ledger(ledger.settings).call_async(
                    "grade", budget, refused, usd_per_hour=limit
                )
        finally:
            release.set()
            await task
        with pytest.raises(HourlySpendCapExceeded):
            await Ledger(ledger.settings).call_async("grade", budget, refused, usd_per_hour=limit)
        refused.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("is_settled", [False, True])
def test_hourly_window_expires_charges_but_keeps_unknown_holds(ledger, monkeypatch, is_settled):
    class Clock(datetime):
        instant = datetime(2026, 1, 1, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.instant

    monkeypatch.setattr("hopper.ledger.datetime", Clock)
    budget = JevBudget("jev-1.14.0", 10_000)
    limit = Decimal("0.00042")
    initial = Clock.instant

    async def first_request():
        Clock.instant = initial + timedelta(hours=2)
        if not is_settled:
            raise ConnectionError("unknown provider outcome")
        return CallResult("yes", JevUsage(10_000))

    first = ledger.call_async("grade", budget, first_request, usd_per_hour=limit)
    if is_settled:
        asyncio.run(first)
    else:
        with pytest.raises(ConnectionError):
            asyncio.run(first)

    restarted = Ledger(ledger.settings)
    refused = AsyncMock(return_value=CallResult("yes", JevUsage(1)))
    with pytest.raises(HourlySpendCapExceeded):
        asyncio.run(restarted.call_async("grade", budget, refused, usd_per_hour=limit))
    refused.assert_not_awaited()

    Clock.instant += timedelta(hours=1, seconds=1)
    next_request = AsyncMock(return_value=CallResult("yes", JevUsage(10_000)))
    if is_settled:
        asyncio.run(restarted.call_async("grade", budget, next_request, usd_per_hour=limit))
        next_request.assert_awaited_once()
    else:
        with pytest.raises(HourlySpendCapExceeded):
            asyncio.run(restarted.call_async("grade", budget, next_request, usd_per_hour=limit))
        next_request.assert_not_awaited()


@pytest.mark.parametrize("blocked_write", [1, 2])
def test_async_cancellation_waits_for_owned_write(ledger, monkeypatch, blocked_write):
    async def run():
        started = asyncio.Event()
        release = Event()
        loop = asyncio.get_running_loop()
        replace = os.replace
        writes = 0

        def pause_replace(source, destination):
            nonlocal writes
            writes += 1
            if writes == blocked_write:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(timeout=5)
            replace(source, destination)

        monkeypatch.setattr("hopper.ledger.os.replace", pause_replace)
        request = AsyncMock(return_value=CallResult("yes", JevUsage(7000)))
        task = asyncio.create_task(
            ledger.call_async(
                "grade", JevBudget("jev-1.14.0", 10_000), request, usd_per_hour=Decimal("1")
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        if blocked_write == 1:
            request.assert_not_awaited()
        else:
            request.assert_awaited_once()

    asyncio.run(run())
    total = Ledger(ledger.settings).summary()[0]
    assert total.held_usd == (Decimal("0.00042") if blocked_write == 1 else 0)
    assert total.settled_usd == (0 if blocked_write == 1 else Decimal("0.000294"))
