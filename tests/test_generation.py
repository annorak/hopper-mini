import json
import multiprocessing
import os
import signal
from decimal import Decimal

import pytest

from hopper import generation
from hopper.config import Settings
from hopper.ledger import Ledger, SpendCapExceeded
from hopper.tasks import load_tasks, validate_tasks


def provider_reply(payload):
    plans = json.loads(payload["messages"][0]["content"])["plans"]
    drafts = []
    for plan in plans:
        key = plan["answer_key"]
        answer = {
            "text": ". ".join(
                ["The requested requirements follow"]
                + key["required_phrases"] + key.get("conflict_documents", [])
            ),
            "citations": key["required_citations"],
            "needs_engineer": key["needs_engineer"],
        }
        if key["mode"] == "calculation":
            answer["value"] = key["value"]
        drafts.append(
            {
                "id": plan["id"],
                "question": f"Please clarify the requirements for {plan['id']}: {plan['brief']}",
                "gold_answer": answer,
            }
        )
    return {
        "id": f"msg-test-{plans[0]['id']}",
        "model": generation.MODEL,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps({"tasks": drafts})}],
        "usage": {"input_tokens": 1000, "output_tokens": 500},
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    settings = Settings(tmp_path / "data", anthropic_api_key="test-key")
    dataset = tmp_path / "datasets" / "rfis.jsonl"
    calls = []

    def provider(settings, endpoint, payload):
        calls.append(endpoint)
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 1000}
        return provider_reply(payload)

    monkeypatch.setattr(generation, "_post_json", provider)
    return settings, dataset, calls


def test_generation_publishes_valid_fixed_dataset_and_second_run_never_calls(setup, monkeypatch):
    settings, dataset, calls = setup
    result = generation.generate_tasks(settings, dataset)
    assert result["generated"] == 150
    assert result["calls"] == 30
    assert result["held_usd"] == "0"
    assert Decimal(result["cost_usd"]) == Decimal("0.21")
    corpus = generation.load_corpus()
    assert validate_tasks(load_tasks(dataset), corpus)["error_count"] == 0
    original = dataset.read_bytes()
    before = list(calls)
    monkeypatch.setattr(
        generation, "_post_json", lambda *args: pytest.fail("Existing tasks must never regenerate")
    )
    reused = generation.generate_tasks(settings, dataset)
    assert reused == {"reused": 150, "generated": 0, "cost_usd": "0", "calls": 0}
    assert dataset.read_bytes() == original
    assert calls == before


def test_rejected_drafts_settle_but_never_publish_partial_dataset(setup, monkeypatch):
    settings, dataset, calls = setup

    def provider(settings, endpoint, payload):
        calls.append(endpoint)
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 1000}
        response = provider_reply(payload)
        response["content"][0]["text"] = "not json"
        return response

    monkeypatch.setattr(generation, "_post_json", provider)
    with pytest.raises(ValueError, match="failed 3 attempts"):
        generation.generate_tasks(settings, dataset)
    assert not dataset.exists()
    summary = Ledger(settings).summary()
    assert sum(row.calls for row in summary) == 3
    assert sum(row.settled_usd for row in summary) == Decimal("0.021")
    assert sum(row.held_usd for row in summary) == 0
    before = list(calls)
    with pytest.raises(ValueError, match="failed 3 attempts"):
        generation.generate_tasks(settings, dataset)
    assert calls == before


def test_checkpoint_directories_are_durable_before_paid_dispatch(setup, monkeypatch):
    settings, dataset, _ = setup
    synced = []
    sync_directory = generation._sync_directory

    def sync(path):
        sync_directory(path)
        synced.append(path)

    def provider(settings, endpoint, payload):
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 1000}
        request = next((settings.data_dir / "corpus" / "generation").glob("*/*.request.json"))
        assert settings.data_dir / "corpus" in synced
        assert request.parent.parent in synced
        assert request.parent in synced
        return provider_reply(payload)

    monkeypatch.setattr(generation, "_sync_directory", sync)
    monkeypatch.setattr(generation, "_post_json", provider)
    assert generation.generate_tasks(settings, dataset)["generated"] == 150


def test_spend_cap_prevents_inference_and_admits_later_safe_retry(setup):
    settings, dataset, calls = setup
    capped = Settings(settings.data_dir, Decimal("0"), anthropic_api_key="test-key")
    with pytest.raises(SpendCapExceeded):
        generation.generate_tasks(capped, dataset)
    assert calls == ["messages/count_tokens"]
    assert not dataset.exists()
    assert generation.generate_tasks(settings, dataset)["generated"] == 150


def test_wrong_model_is_charged_and_rejected(setup, monkeypatch):
    settings, dataset, calls = setup

    def provider(settings, endpoint, payload):
        calls.append(endpoint)
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 1000}
        response = provider_reply(payload)
        response["model"] = "different-model"
        return response

    monkeypatch.setattr(generation, "_post_json", provider)
    with pytest.raises(ValueError, match="Provider returned"):
        generation.generate_tasks(settings, dataset)
    assert not dataset.exists()
    assert sum(row.settled_usd for row in Ledger(settings).summary()) == Decimal("0.007")


def test_lost_attempt_file_cannot_override_existing_ledger_entry(setup, monkeypatch):
    settings, dataset, _ = setup
    generation.generate_tasks(settings, dataset)
    dataset.unlink()
    attempt = next((settings.data_dir / "corpus" / "generation").glob("*/*.request.json"))
    attempt.unlink()
    monkeypatch.setattr(
        generation, "_post_json", lambda *args: pytest.fail("Ledger entry prevents redispatch")
    )
    with pytest.raises(RuntimeError, match="Unresolved generation attempt"):
        generation.generate_tasks(settings, dataset)


def _kill_during_provider(settings, dataset, ready):
    def provider(settings, endpoint, payload):
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 1000}
        ready.send("dispatched")
        signal.pause()
        raise AssertionError("Expected SIGKILL")

    generation._post_json = provider
    generation.generate_tasks(settings, dataset)


def test_sigkill_after_dispatch_keeps_hold_and_restart_does_not_retry(setup, monkeypatch):
    settings, dataset, _ = setup
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_kill_during_provider, args=(settings, dataset, sending))
    process.start()
    sending.close()
    try:
        assert receiving.poll(15), "Generation did not reach provider dispatch"
        assert receiving.recv() == "dispatched"
        os.kill(process.pid, signal.SIGKILL)
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        receiving.close()
    monkeypatch.setattr(
        generation, "_post_json", lambda *args: pytest.fail("Uncertain call must not retry")
    )
    with pytest.raises(RuntimeError, match="Unresolved generation attempt"):
        generation.generate_tasks(settings, dataset)
    assert sum(row.held_usd for row in Ledger(settings).summary()) > 0
    assert not dataset.exists()


def _kill_before_publication(settings, dataset, ready):
    original = generation._publish

    def publish(path, content, temporary):
        if path == dataset:
            ready.send("ready to publish")
            signal.pause()
            raise AssertionError("Expected SIGKILL")
        original(path, content, temporary)

    def provider(settings, endpoint, payload):
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 1000}
        return provider_reply(payload)

    generation._publish = publish
    generation._post_json = provider
    generation.generate_tasks(settings, dataset)


def test_sigkill_before_dataset_publication_resumes_saved_responses(setup, monkeypatch):
    settings, dataset, _ = setup
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_kill_before_publication, args=(settings, dataset, sending))
    process.start()
    sending.close()
    try:
        assert receiving.poll(30), "Generation did not reach dataset publication"
        assert receiving.recv() == "ready to publish"
        os.kill(process.pid, signal.SIGKILL)
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        receiving.close()
    assert not dataset.exists()
    before = Ledger(settings).path.read_bytes()
    monkeypatch.setattr(
        generation, "_post_json", lambda *args: pytest.fail("Saved responses must not redraft")
    )
    assert generation.generate_tasks(settings, dataset)["generated"] == 150
    assert Ledger(settings).path.read_bytes() == before
