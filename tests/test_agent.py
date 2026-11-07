import base64
import copy
import http.client
import io
import json
import shutil
import threading
import urllib.error
from dataclasses import asdict, replace
from decimal import Decimal

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from typer.testing import CliRunner

from hopper import agent as agent_module
from hopper import otel
from hopper.agent import Agent, Conversation, ProviderError, RunResult, Task
from hopper.cli import app
from hopper.config import Settings, TokenBudget, TokenUsage, token_cost
from hopper.corpus import build_corpus, load_corpus
from hopper.ledger import CallResult

REAL_TOOL_FAULT = agent_module.tool_fault
USAGE = TokenUsage(100, 20, 50, 10)
ANSWER = {
    "text": "Follow the governing specification.",
    "citations": ["SPEC-01/1.1"],
    "needs_engineer": False,
}
FINDINGS = {
    "findings": "The specification governs.",
    "citations": ["SPEC-01/1.1"],
    "needs_engineer": False,
}
TRIAGE = {"rfi_type": "clarification", "mentioned_ids": ["SPEC-01"], "needs_calculation": False}


def tool(name, arguments, identifier="tool-1"):
    return {"type": "tool_use", "id": identifier, "name": name, "input": arguments}


def response(content, stop="tool_use"):
    return {
        "id": "msg-fake",
        "model": "claude-haiku-5-5",
        "stop_reason": stop,
        "content": content,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 10,
        },
    }


def triage(value=TRIAGE):
    return response([{"type": "text", "text": json.dumps(value)}], "end_turn")


def findings():
    return response([tool("report_findings", FINDINGS)])


def answer():
    return response([tool("submit_answer", ANSWER)])


class Provider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def __call__(self, settings, endpoint, payload):
        self.requests.append((endpoint, copy.deepcopy(payload)))
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 100}
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return copy.deepcopy(reply)

    @property
    def paid(self):
        return [payload for endpoint, payload in self.requests if endpoint == "messages"]


@pytest.fixture(scope="session")
def rendered_corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("agent-corpus")
    build_corpus(directory)
    return directory


@pytest.fixture
def environment(tmp_path, rendered_corpus, monkeypatch):
    shutil.copytree(rendered_corpus / "corpus", tmp_path / "corpus")
    settings = Settings(tmp_path, anthropic_api_key="test-key")
    exporters = []

    def exporter(**kwargs):
        result = InMemorySpanExporter()
        exporters.append(result)
        return result

    monkeypatch.setattr(otel, "OTLPSpanExporter", exporter)
    monkeypatch.setattr(agent_module, "tool_fault", lambda *args: False)
    instance = Agent(settings)
    task = Task("RFI-TEST", "Which requirement governs?", load_corpus().version)
    return instance, task, exporters


def install_provider(monkeypatch, replies):
    provider = Provider(replies)
    monkeypatch.setattr(agent_module, "_post_json", provider)
    return provider


def exported(exporters):
    return [span for exporter in exporters for span in exporter.get_finished_spans()]


def test_count_tokens_retries_rate_limit_but_paid_request_does_not(monkeypatch, tmp_path):
    calls, waits = [], []

    def send(request, timeout):
        calls.append(request.full_url)
        if len(calls) != 2:
            raise urllib.error.HTTPError(
                request.full_url, 429, "rate limit", {"retry-after": "7"}, io.BytesIO(b"")
            )
        return io.BytesIO(b'{"input_tokens":123}')

    monkeypatch.setattr(agent_module.urllib.request, "urlopen", send)
    monkeypatch.setattr(agent_module, "sleep", waits.append)
    settings = Settings(tmp_path, anthropic_api_key="test")
    assert agent_module._count_tokens(settings, {}) == 123
    assert len(calls) == 2 and waits == [7]
    with pytest.raises(agent_module.ProviderRateLimited):
        agent_module._post_json(settings, "messages", {})
    assert len(calls) == 3 and waits == [7]


def test_count_token_rate_retries_are_bounded(monkeypatch, tmp_path):
    calls, waits = [], []

    def send(*args):
        calls.append(args)
        raise agent_module.ProviderRateLimited("messages/count_tokens", 1)

    monkeypatch.setattr(agent_module, "_post_json", send)
    monkeypatch.setattr(agent_module, "sleep", waits.append)
    with pytest.raises(agent_module.ProviderRateLimited):
        agent_module._count_tokens(Settings(tmp_path), {})
    assert len(calls) == 6 and waits == [2, 4, 8, 16, 32]


def test_provider_outage_cancels_queued_conversations(environment, monkeypatch):
    instance, task, _ = environment
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise ProviderError("token counter unavailable")

    monkeypatch.setattr(agent_module, "_post_json", unavailable)
    tasks = [replace(task, id=f"task-{index}") for index in range(100)]
    with pytest.raises(agent_module.AgentError, match="Batch stopped"):
        instance.batch(tasks, 1)
    assert 1 <= len(calls) <= 4
    assert instance.ledger.summary() == []


def test_seed_identity_and_cost_stage_cover_every_emitted_span(environment, monkeypatch):
    instance, task, exporters = environment
    instance = Agent(instance.settings, seed_id="seed")
    install_provider(monkeypatch, [triage(), findings(), answer()])
    result = instance.run(task)
    assert result.answer == ANSWER
    spans = exported(exporters)
    assert all(span.resource.attributes["hopper.seed.id"] == "seed" for span in spans)
    root = next(span for span in spans if span.parent is None)
    assert root.attributes["hopper.research.step_cap"] == instance.baseline.research_step_cap
    assert all(total.stage.startswith("seed:seed:") for total in instance.ledger.summary())


def test_transient_fault_kinds_are_real_error_spans_and_reporting_still_succeeds(
    environment, monkeypatch
):
    instance, task, exporters = environment
    monkeypatch.setattr(agent_module, "tool_fault", lambda *args: True)
    install_provider(
        monkeypatch,
        [
            triage(),
            *[response([tool("read_section", {"id": "SPEC-01"}, f"read-{i}")]) for i in range(3)],
            findings(),
            answer(),
        ],
    )
    assert instance.run(task).answer == ANSWER
    errors = [span for span in exported(exporters) if span.status.status_code == StatusCode.ERROR]
    assert {span.attributes["error.type"] for span in errors} == {
        "ToolUnavailable",
        "ToolTimedOut",
        "ToolRateLimited",
    }
    assert len(errors) == 3


def test_complete_exchange_has_exact_stage_tools_and_settled_usage(environment, monkeypatch):
    instance, task, exporters = environment
    provider = install_provider(monkeypatch, [triage(), findings(), answer()])
    result = instance.run(task)
    assert result.answer == ANSWER
    assert result.error is None
    assert result.held_usd == "0"
    assert Decimal(result.settled_usd) == token_cost("claude-haiku-5-5", USAGE) * 3
    assert len(provider.paid) == 3
    assert "tools" not in provider.paid[0]
    assert {item["name"] for item in provider.paid[1]["tools"]} == {
        "search_documents",
        "read_section",
        "list_addenda",
        "read_drawing",
        "calculate",
        "report_findings",
    }
    assert [item["name"] for item in provider.paid[2]["tools"]] == ["submit_answer"]
    assert all(span.status.status_code != StatusCode.ERROR for span in exported(exporters))


def test_model_options_cache_and_signed_thinking_survive_tool_replay(environment, monkeypatch):
    instance, task, exporters = environment
    thinking = {
        "type": "thinking",
        "thinking": "I should read the source.",
        "signature": "signed-provider-thinking",
    }
    provider = install_provider(
        monkeypatch,
        [
            triage(),
            response([thinking, tool("read_section", {"id": "SPEC-01/1.1"})]),
            findings(),
            answer(),
        ],
    )
    assert instance.run(task).answer == ANSWER
    for payload in provider.paid:
        assert payload["model"] == "claude-haiku-5-5"
        assert payload["max_tokens"] == 16000
        assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert payload["output_config"]["effort"] == "low"
        assert not {"temperature", "top_p", "top_k"} & payload.keys()
        assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    replay = provider.paid[2]["messages"]
    assert thinking in next(item["content"] for item in replay if item["role"] == "assistant")
    outputs = [
        json.loads(span.attributes["gen_ai.output.messages"])
        for span in exported(exporters)
        if "gen_ai.output.messages" in span.attributes
    ]
    assert any(
        part.get("type") == "reasoning" and part.get("content") == thinking["thinking"]
        for messages in outputs
        for message in messages
        for part in message["parts"]
    )


def test_trace_tree_has_task_identity_and_nested_research_tools(environment, monkeypatch):
    instance, task, exporters = environment
    install_provider(monkeypatch, [triage(), findings(), answer()])
    result = instance.run(task, repeat=7)
    spans = exported(exporters)
    root = next(span for span in spans if span.parent is None)
    research = next(span for span in spans if span.name == "invoke_agent research")
    assert root.name == "invoke_agent rfi"
    assert format(root.context.trace_id, "032x") == result.trace_id
    assert research.parent.span_id == root.context.span_id
    assert any(
        span.parent
        and span.parent.span_id == research.context.span_id
        and span.attributes["gen_ai.operation.name"] == "chat"
        for span in spans
    )
    assert any(
        span.parent
        and span.parent.span_id == research.context.span_id
        and span.attributes["gen_ai.operation.name"] == "execute_tool"
        for span in spans
    )
    assert all(span.resource.attributes["hopper.task.id"] == task.id for span in spans)
    assert all(span.resource.attributes["service.name"] == "rfi-agent" for span in spans)
    assert all(
        span.resource.attributes["hopper.agent.version"] == instance.baseline.version
        for span in spans
    )
    assert root.attributes["hopper.repeat"] == 7


def test_drawing_sends_png_but_all_trace_attributes_use_uri(environment, monkeypatch):
    instance, task, exporters = environment
    drawing_id = next(iter(instance.images))
    provider = install_provider(
        monkeypatch,
        [triage(), response([tool("read_drawing", {"id": drawing_id})]), findings(), answer()],
    )
    assert instance.run(task).answer == ANSWER
    research_messages = provider.paid[2]["messages"]
    tool_result = next(
        block
        for message in research_messages
        for block in message["content"]
        if block["type"] == "tool_result"
    )
    image = next(block for block in tool_result["content"] if block["type"] == "image")
    png = base64.b64decode(image["source"]["data"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert png == instance.images[drawing_id].png
    attributes = json.dumps([dict(span.attributes) for span in exported(exporters)])
    assert f"corpus://drawings/{drawing_id}.png" in attributes
    assert image["source"]["data"] not in attributes
    assert '"base64"' not in attributes


def test_findings_require_cited_drawings_to_have_been_read_as_images(environment, monkeypatch):
    instance, task, exporters = environment
    drawing_findings = {**FINDINGS, "citations": ["M-101/1"]}
    provider = install_provider(
        monkeypatch,
        [
            triage(),
            response([tool("report_findings", drawing_findings)]),
            response([tool("read_drawing", {"id": "M-101"})]),
            response([tool("report_findings", drawing_findings)]),
            answer(),
        ],
    )
    result = instance.run(task)
    assert result.answer == ANSWER
    rejected = provider.paid[2]["messages"][-1]["content"][0]
    assert rejected["is_error"] is True
    assert "read_drawing" in rejected["content"][0]["text"]
    image = provider.paid[3]["messages"][-1]["content"][0]["content"][1]
    assert image["type"] == "image"
    assert base64.b64decode(image["source"]["data"]) == instance.images["M-101"].png


def test_cap_blocks_paid_request_before_provider_call(environment, monkeypatch):
    instance, task, exporters = environment
    instance.settings = replace(instance.settings, spend_cap_usd=Decimal("0"))
    instance.ledger.settings = instance.settings
    provider = install_provider(monkeypatch, [triage()])
    result = instance.run(task)
    assert result.answer is None
    assert "SpendCapExceeded" in result.error
    assert provider.paid == []
    assert result.settled_usd == result.held_usd == "0"
    assert any(span.status.status_code == StatusCode.ERROR for span in exported(exporters))


def test_unknown_provider_outcome_retains_reservation_and_is_not_retried(environment, monkeypatch):
    instance, task, exporters = environment
    provider = install_provider(monkeypatch, [ProviderError("outcome unknown")])
    result = instance.run(task)
    assert result.answer is None
    assert "ProviderError" in result.error
    assert len(provider.paid) == 1
    assert Decimal(result.held_usd) > 0
    assert result.settled_usd == "0"
    assert len(instance.ledger.summary()) == 1
    assert any(span.status.status_code == StatusCode.ERROR for span in exported(exporters))


def test_truncated_http_body_reports_failure_and_retains_paid_reservation(environment, monkeypatch):
    instance, task, _ = environment
    post = agent_module._post_json
    dispatched = []

    class TruncatedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            raise http.client.IncompleteRead(b'{"usage":', 100)

    def send(request, timeout):
        dispatched.append(request)
        return TruncatedResponse()

    def provider(settings, endpoint, payload):
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 100}
        return post(settings, endpoint, payload)

    monkeypatch.setattr(agent_module.urllib.request, "urlopen", send)
    monkeypatch.setattr(agent_module, "_post_json", provider)
    result = instance.run(task)
    assert result.answer is None
    assert "ProviderError" in result.error
    assert len(dispatched) == 1
    assert len(result.trace_id) == 32
    assert Decimal(result.held_usd) > 0
    assert result.settled_usd == "0"


def test_real_http_adapter_encodes_requests_and_decodes_replies(environment, monkeypatch):
    instance, task, _ = environment
    replies = iter([triage(), findings(), answer()])
    paid = []

    def send(request, timeout):
        assert timeout == 180
        assert request.get_header("X-api-key") == "test-key"
        payload = json.loads(request.data)
        if request.full_url.endswith("/count_tokens"):
            reply = {"input_tokens": 100}
        else:
            paid.append(payload)
            reply = next(replies)
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(agent_module.urllib.request, "urlopen", send)
    result = instance.run(task)
    assert result.answer == ANSWER
    assert result.error is None
    assert len(paid) == 3
    assert all(payload["model"] == "claude-haiku-5-5" for payload in paid)


def test_interrupted_batch_cancels_pending_conversations(environment, monkeypatch):
    instance, task, _ = environment
    entered = threading.Barrier(4)
    started = []
    lock = threading.Lock()

    def run(task, repeat, conversation):
        with lock:
            started.append(task.id)
            is_first_wave = len(started) <= 4
        if is_first_wave:
            entered.wait(timeout=5)
        if task.id != "first":
            assert conversation.cancelled.wait(timeout=5)
        return RunResult(task.id, repeat, 0, conversation.id, "trace", ANSWER, None, "0", "0", 0)

    def interrupt(result):
        raise KeyboardInterrupt

    monkeypatch.setattr(instance, "run", run)
    tasks = [replace(task, id=identifier) for identifier in ["first", *map(str, range(9))]]
    with pytest.raises(KeyboardInterrupt):
        instance.batch(tasks, 1, followups=False, progress=interrupt)
    assert 4 <= len(started) < len(tasks)


def test_cancelled_conversation_makes_no_provider_request(environment, monkeypatch):
    instance, task, _ = environment
    conversation = Conversation()
    conversation.cancelled.set()
    provider = install_provider(monkeypatch, [])
    result = instance.run(task, conversation=conversation)
    assert result.answer is None
    assert "cancelled" in result.error
    assert provider.requests == []
    assert result.settled_usd == result.held_usd == "0"


def test_interrupt_during_submission_cancels_running_conversation(environment, monkeypatch):
    instance, task, _ = environment
    entered, cancelled = threading.Event(), threading.Event()
    submit = agent_module.ThreadPoolExecutor.submit

    def run(task, repeat, conversation):
        entered.set()
        if conversation.cancelled.wait(timeout=5):
            cancelled.set()
        return RunResult(task.id, repeat, 0, conversation.id, "trace", ANSWER, None, "0", "0", 0)

    def interrupt_submission(pool, *args, **kwargs):
        if entered.is_set():
            raise KeyboardInterrupt
        future = submit(pool, *args, **kwargs)
        assert entered.wait(timeout=5)
        return future

    monkeypatch.setattr(instance, "run", run)
    monkeypatch.setattr(agent_module.ThreadPoolExecutor, "submit", interrupt_submission)
    with pytest.raises(KeyboardInterrupt):
        instance.batch([task, replace(task, id="second")], 1, followups=False)
    assert cancelled.is_set()


def test_cancellation_during_reservation_settles_unsent_request_at_zero(environment, monkeypatch):
    instance, task, _ = environment
    conversation = Conversation()
    write = instance.ledger._write
    provider = install_provider(monkeypatch, [triage(), findings(), answer()])

    def cancel_after_reservation(rows):
        write(rows)
        if rows[-1]["state"] == "reserved":
            conversation.cancelled.set()

    monkeypatch.setattr(instance.ledger, "_write", cancel_after_reservation)
    result = instance.run(task, conversation=conversation)
    assert result.answer is None
    assert "cancelled" in result.error
    assert [endpoint for endpoint, _ in provider.requests] == ["messages/count_tokens"]
    assert Decimal(result.settled_usd) == Decimal(result.held_usd) == 0
    assert instance.ledger._read()[0]["state"] == "settled"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_provider_stop_has_no_answer_and_records_error(environment, monkeypatch, stop):
    instance, task, exporters = environment
    provider = install_provider(monkeypatch, [response([], stop)])
    result = instance.run(task)
    assert result.answer is None
    assert stop in result.error
    assert len(provider.paid) == 1
    assert Decimal(result.settled_usd) > 0
    assert result.held_usd == "0"
    assert any(span.status.status_code == StatusCode.ERROR for span in exported(exporters))


@pytest.mark.parametrize(
    "value", [[], {**TRIAGE, "answer_key": "hidden"}, {**TRIAGE, "needs_calculation": "false"}]
)
def test_malformed_triage_stops_before_research(environment, monkeypatch, value):
    instance, task, _ = environment
    provider = install_provider(monkeypatch, [triage(value)])
    result = instance.run(task)
    assert result.answer is None and result.error
    assert len(provider.paid) == 1


@pytest.mark.parametrize(
    "blocks",
    [
        [{"type": "text", "text": "I cannot answer."}],
        [tool("read_section", {"id": "SPEC-01"}), tool("list_addenda", {}, "tool-2")],
        [tool("submit_answer", ANSWER)],
    ],
)
def test_research_requires_exactly_one_research_tool(environment, monkeypatch, blocks):
    instance, task, _ = environment
    provider = install_provider(monkeypatch, [triage(), response(blocks)])
    result = instance.run(task)
    assert result.answer is None and result.error
    assert len(provider.paid) == 2


def test_invalid_tool_arguments_are_visible_to_model_and_can_be_corrected(environment, monkeypatch):
    instance, task, exporters = environment
    provider = install_provider(
        monkeypatch,
        [
            triage(),
            response([tool("read_section", {"id": "DOES-NOT-EXIST"})]),
            findings(),
            answer(),
        ],
    )
    result = instance.run(task)
    assert result.answer == ANSWER
    results = [
        block
        for message in provider.paid[2]["messages"]
        for block in message["content"]
        if block["type"] == "tool_result"
    ]
    assert len(results) == 1
    assert results[0]["is_error"] is True
    assert "DOES-NOT-EXIST" in results[0]["content"][0]["text"]
    tool_span = next(
        span for span in exported(exporters) if span.name == "execute_tool read_section"
    )
    assert tool_span.status.status_code == StatusCode.ERROR


@pytest.mark.parametrize(
    "blocks",
    [
        [{"type": "refusal", "refusal": "No"}],
        [tool("submit_answer", {**ANSWER, "citations": ["bogus"]})],
    ],
)
def test_writer_refusal_or_invalid_answer_never_returns_answer(environment, monkeypatch, blocks):
    instance, task, exporters = environment
    install_provider(monkeypatch, [triage(), findings(), response(blocks, "end_turn")])
    result = instance.run(task)
    assert result.answer is None and result.error
    assert any(
        span.name == "invoke_agent rfi" and span.status.status_code == StatusCode.ERROR
        for span in exported(exporters)
    )


def test_research_stops_after_eight_model_calls_without_writer(environment, monkeypatch):
    instance, task, _ = environment
    instance.baseline = replace(instance.baseline, research_step_cap=8)
    provider = install_provider(
        monkeypatch,
        [
            triage(),
            *[
                response([tool("read_section", {"id": "SPEC-01"}, f"read-{index}")])
                for index in range(8)
            ],
        ],
    )
    result = instance.run(task)
    assert result.answer is None
    assert "cap" in result.error.lower()
    assert len(provider.paid) == 9
    assert all(payload["tools"][0]["name"] != "submit_answer" for payload in provider.paid[1:])


def test_seeded_fault_is_returned_then_model_can_retry(environment, monkeypatch):
    instance, task, exporters = environment
    task = replace(
        task,
        id=next(
            f"RFI-{index}"
            for index in range(10000)
            if REAL_TOOL_FAULT(f"RFI-{index}", 0, 1)
            and not any(REAL_TOOL_FAULT(f"RFI-{index}", 0, number) for number in (2, 3, 4))
        ),
    )
    monkeypatch.setattr(agent_module, "tool_fault", REAL_TOOL_FAULT)
    provider = install_provider(
        monkeypatch,
        [
            triage(),
            response([tool("read_section", {"id": "SPEC-01"}, "first")]),
            response([tool("read_section", {"id": "SPEC-01"}, "retry")]),
            findings(),
            answer(),
        ],
    )
    result = instance.run(task)
    assert result.answer == ANSWER
    retry_input = provider.paid[2]["messages"][-1]["content"][0]
    assert retry_input["is_error"] is True
    assert "retry" in retry_input["content"][0]["text"].lower()
    recovered_input = provider.paid[3]["messages"][-1]["content"][0]
    assert recovered_input.get("is_error") is not True
    assert sum(span.status.status_code == StatusCode.ERROR for span in exported(exporters)) == 1


def test_followup_has_new_trace_same_conversation_and_prior_answer(environment, monkeypatch):
    instance, task, exporters = environment
    provider = install_provider(monkeypatch, [triage(), findings(), answer()] * 2)
    conversation = Conversation()
    first = instance.run(task, conversation=conversation)
    second = instance.run(
        replace(task, question="Please clarify the field instruction."), conversation=conversation
    )
    assert first.answer == second.answer == ANSWER
    assert first.trace_id != second.trace_id
    assert first.conversation_id == second.conversation_id
    assert [first.turn, second.turn] == [0, 1]
    for request in (provider.paid[0], provider.paid[3]):
        output = request["output_config"]["format"]
        assert output["type"] == "json_schema"
        assert output["schema"] == {
            "type": "object",
            "properties": {
                "rfi_type": {"type": "string"},
                "mentioned_ids": {"type": "array", "items": {"type": "string"}},
                "needs_calculation": {"type": "boolean"},
            },
            "required": ["rfi_type", "mentioned_ids", "needs_calculation"],
            "additionalProperties": False,
        }
    assert all(
        "format" not in request["output_config"] for request in provider.paid if "tools" in request
    )
    assert provider.paid[3]["messages"][:2] == [
        otel.text_message("user", task.question),
        otel.text_message("assistant", json.dumps(ANSWER)),
    ]
    roots = [span for span in exported(exporters) if span.parent is None]
    assert len(roots) == 2
    assert {span.attributes["gen_ai.conversation.id"] for span in roots} == {conversation.id}


def test_four_worker_batch_costs_exclude_other_runs_and_are_not_double_counted(
    environment, monkeypatch
):
    instance, task, exporters = environment
    instance.ledger.call(
        "existing-other-work",
        TokenBudget("claude-haiku-5-5", 200, 30),
        lambda: CallResult("prior", USAGE),
    )
    barrier = threading.Barrier(4)

    def provider(settings, endpoint, payload):
        if endpoint == "messages/count_tokens":
            return {"input_tokens": 100}
        names = [item["name"] for item in payload.get("tools", [])]
        if not names:
            barrier.wait(timeout=10)
            return triage()
        return answer() if names == ["submit_answer"] else findings()

    monkeypatch.setattr(agent_module, "_post_json", provider)
    tasks = [replace(task, id=f"RFI-BATCH-{index}") for index in range(4)]
    progress = []
    result = instance.batch(tasks, 1, followups=False, progress=progress.append)
    call_cost = token_cost("claude-haiku-5-5", USAGE)
    assert result["base_runs"] == result["answered"] == result["runs"] == 4
    assert result["followup_runs"] == result["without_answer"] == result["errors"] == 0
    assert result["held_usd"] == "0"
    assert len(progress) == 4
    assert all(Decimal(item["settled_usd"]) == call_cost * 3 for item in result["results"])
    assert Decimal(result["settled_usd"]) == call_cost * 12
    assert sum(total.settled_usd for total in instance.ledger.summary()) == call_cost * 13
    roots = [span for span in exported(exporters) if span.parent is None]
    assert len({span.context.trace_id for span in roots}) == 4
    assert {span.resource.attributes["hopper.task.id"] for span in roots} == {t.id for t in tasks}


def test_batch_counts_scheduled_followups_separately(environment, monkeypatch):
    instance, task, _ = environment
    monkeypatch.setattr(agent_module, "followup_count", lambda *args: 2)
    install_provider(monkeypatch, [triage(), findings(), answer()] * 3)
    result = instance.batch([task], 1)
    assert result["base_runs"] == 1
    assert result["followup_runs"] == 2
    assert result["answered"] == 3
    assert [item["turn"] for item in result["results"]] == [0, 1, 2]
    assert len({item["trace_id"] for item in result["results"]}) == 3
    assert len({item["conversation_id"] for item in result["results"]}) == 1


def test_task_projection_keeps_gold_metadata_out_of_paid_requests(environment, monkeypatch):
    instance, task, _ = environment
    record = {
        **asdict(task),
        **{
            name: f"SECRET-{name}"
            for name in ("mode", "answer_key", "gold_answer", "cases", "generation")
        },
    }
    projected = Task.from_record(record)
    assert asdict(projected) == asdict(task)
    provider = install_provider(monkeypatch, [triage(), findings(), answer()])
    assert instance.run(projected).answer == ANSWER
    wire = json.dumps(provider.paid)
    assert all(value not in wire for key, value in record.items() if key not in asdict(task))


@pytest.mark.parametrize(
    "args", [["agent", "--help"], ["agent", "run", "--help"], ["agent", "batch", "--help"]]
)
def test_agent_cli_help_does_not_need_api_or_corpus(args):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_agent_cli_projects_task_and_prints_result_json(environment, monkeypatch):
    from hopper import tasks

    instance, task, _ = environment
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: instance.settings))
    monkeypatch.setattr(
        tasks, "load_tasks", lambda path: [{**asdict(task), "gold_answer": "secret"}]
    )
    install_provider(monkeypatch, [triage(), findings(), answer()])
    result = CliRunner().invoke(app, ["agent", "run", "--task", task.id])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["answer"] == ANSWER
    assert len(payload["trace_id"]) == 32
    assert payload["error"] is None


def test_agent_cli_failed_run_prints_json_and_exits_nonzero(environment, monkeypatch):
    from hopper import tasks

    instance, task, _ = environment
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: instance.settings))
    monkeypatch.setattr(tasks, "load_tasks", lambda path: [asdict(task)])
    install_provider(monkeypatch, [response([], "refusal")])
    result = CliRunner().invoke(app, ["agent", "run", "--task", task.id])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["answer"] is None
    assert "refusal" in payload["error"]
