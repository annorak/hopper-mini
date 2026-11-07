import hashlib
import http.client
import json
import logging
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from threading import Event
from time import perf_counter, sleep
from uuid import uuid4

from hopper import otel
from hopper.calculator import calculate
from hopper.config import Settings, TokenBudget, TokenUsage
from hopper.corpus import ADDENDUM_IDS, Corpus, load_corpus, search_corpus
from hopper.ledger import CallResult, Ledger

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16_000
TOKEN_COUNT_ATTEMPTS = 6
DEFAULT_BASELINE = Path(__file__).resolve().parents[2] / "agents" / "rfi" / "baseline.json"
FOLLOWUP_QUESTIONS = (
    "Please restate the governing requirement and its exact source citations for our field team.",
    "What uncertainty still needs a decision from the engineer of record? Cite the sources.",
)
TOOL_FIELDS = {
    "search_documents": {"query": "string"},
    "read_section": {"id": "string"},
    "list_addenda": {},
    "read_drawing": {"id": "string"},
    "calculate": {"code": "string"},
    "report_findings": {"findings": "string", "citations": "array", "needs_engineer": "boolean"},
    "submit_answer": {"text": "string", "citations": "array", "needs_engineer": "boolean"},
}


class AgentError(RuntimeError):
    pass


class ProviderError(AgentError):
    pass


class ProviderRateLimited(ProviderError):
    def __init__(self, endpoint: str, retry_after: float):
        super().__init__(f"Anthropic {endpoint} returned HTTP 429; retry after {retry_after:g}s")
        self.retry_after = retry_after


class ModelStopped(AgentError):
    pass


class ToolUnavailable(AgentError):
    pass


class ToolTimedOut(AgentError):
    pass


class ToolRateLimited(AgentError):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    corpus_version: str

    @classmethod
    def from_record(cls, record: dict) -> "Task":
        return cls(*(record[key] for key in ("id", "question", "corpus_version")))


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid4().hex)
    history: list[dict] = field(default_factory=list)
    tool_number: int = 0
    turn: int = 0
    cancelled: Event = field(default_factory=Event, repr=False)


@dataclass(frozen=True)
class RunResult:
    task_id: str
    repeat: int
    turn: int
    conversation_id: str
    trace_id: str
    answer: dict | None
    error: str | None
    settled_usd: str
    held_usd: str
    elapsed_seconds: float
    error_type: str | None = None

    @property
    def should_stop_batch(self) -> bool:
        return (
            self.error_type
            in {
                "ProviderError",
                "ProviderRateLimited",
                "SpendCapExceeded",
                "UsageLimitExceeded",
            }
            or Decimal(self.held_usd) > 0
            or self.answer is not None
            and self.error is not None
        )


@dataclass(frozen=True)
class Baseline:
    prompts: dict[str, str]
    descriptions: dict[str, str]
    research_step_cap: int
    effort: str
    version: str

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        data = json.loads(path.read_text())
        if data.keys() != {"prompts", "tool_descriptions", "research_step_cap", "effort"}:
            raise ValueError(
                "Baseline must contain prompts, tool_descriptions, research_step_cap, effort"
            )
        prompts, descriptions = data["prompts"], data["tool_descriptions"]
        if not isinstance(prompts, dict) or prompts.keys() != {
            "shared",
            "triage",
            "research",
            "writer",
        }:
            raise ValueError("Baseline prompts must contain shared, triage, research, writer")
        if not isinstance(descriptions, dict) or descriptions.keys() != TOOL_FIELDS.keys():
            raise ValueError("Baseline must describe exactly the seven RFI tools")
        if any(
            not isinstance(text, str) or not text.strip()
            for text in [*prompts.values(), *descriptions.values()]
        ):
            raise ValueError("Baseline prompts and tool descriptions must be nonempty text")
        if type(data["research_step_cap"]) is not int or not 1 <= data["research_step_cap"] <= 12:
            raise ValueError("research_step_cap must be an integer from 1 to 12")
        if data["effort"] not in {"low", "medium", "high"}:
            raise ValueError("effort must be low, medium, or high")
        version = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return cls(prompts, descriptions, data["research_step_cap"], data["effort"], version)

    def tools(self, names: list[str]) -> list[dict]:
        definitions = []
        for name in names:
            properties = {key: {"type": kind} for key, kind in TOOL_FIELDS[name].items()}
            if "citations" in properties:
                properties["citations"]["items"] = {"type": "string"}
            if name == "submit_answer":
                properties["value"] = {"type": "number"}
            definitions.append(
                {
                    "name": name,
                    "description": self.descriptions[name],
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": list(TOOL_FIELDS[name]),
                        "additionalProperties": False,
                    },
                }
            )
        return definitions


def _post_json(settings: Settings, endpoint: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"https://api.anthropic.com/v1/{endpoint}",
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        error.close()
        if error.code == 429:
            try:
                delay = float((error.headers or {}).get("retry-after", "10"))
            except ValueError:
                delay = math.inf
            if not math.isfinite(delay) or not 0 <= delay <= 60:
                raise ProviderError(
                    "Rate-limit wait exceeds the bounded 60-second retry delay"
                ) from error
            raise ProviderRateLimited(endpoint, delay) from error
        raise ProviderError(
            f"Anthropic {endpoint} returned HTTP {error.code}; not retried"
        ) from error
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise ProviderError(f"Anthropic {endpoint} outcome is unknown; not retried") from error


def _count_tokens(settings: Settings, payload: dict) -> int:
    for attempt in range(TOKEN_COUNT_ATTEMPTS):
        try:
            return _post_json(settings, "messages/count_tokens", payload)["input_tokens"]
        except ProviderRateLimited as error:
            if attempt == TOKEN_COUNT_ATTEMPTS - 1:
                raise
            delay = max(error.retry_after, 2 ** (attempt + 1))
            logging.getLogger(__name__).warning(
                "Token counting rate limited; retry %s/%s in %ss",
                attempt + 1,
                TOKEN_COUNT_ATTEMPTS - 1,
                delay,
            )
            sleep(delay)


def _usage(response: dict) -> TokenUsage:
    usage = response["usage"]
    return TokenUsage(
        usage["input_tokens"],
        usage["output_tokens"],
        usage.get("cache_read_input_tokens", 0),
        usage.get("cache_creation_input_tokens", 0),
    )


def tool_fault(task_id: str, repeat: int, call_number: int) -> bool:
    digest = hashlib.sha256(f"{task_id}:{repeat}:{call_number}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < 3


def followup_count(task_id: str, repeat: int) -> int:
    digest = hashlib.sha256(f"followup:{task_id}:{repeat}".encode()).digest()
    return 1 + digest[8] % 2 if int.from_bytes(digest[:8], "big") % 5 == 0 else 0


def _arguments(name: str, arguments: dict, corpus: Corpus) -> None:
    if name not in TOOL_FIELDS or not isinstance(arguments, dict):
        raise ValueError("Unknown tool or non-object tool arguments")
    fields = TOOL_FIELDS[name]
    optional = {"value"} if name == "submit_answer" else set()
    if not fields.keys() <= arguments.keys() or arguments.keys() - fields.keys() - optional:
        raise ValueError(f"Invalid argument fields for {name}")
    for key, kind in fields.items():
        value = arguments[key]
        if kind == "string" and (
            not isinstance(value, str) or not value.strip() or len(value) > 32_768
        ):
            raise ValueError(f"{name}.{key} must be nonempty text of at most 32768 characters")
        if kind == "boolean" and type(value) is not bool:
            raise ValueError(f"{name}.{key} must be boolean")
        if kind == "array" and (
            not isinstance(value, list)
            or len(value) > 100
            or any(
                not isinstance(item, str)
                or item not in corpus.sections
                and item not in corpus.documents
                for item in value
            )
        ):
            raise ValueError(f"{name}.{key} must contain known source citation IDs")
    if "value" in arguments and (
        type(arguments["value"]) not in (int, float) or not math.isfinite(arguments["value"])
    ):
        raise ValueError("submit_answer.value must be a finite number")


class Agent:
    def __init__(
        self, settings: Settings, baseline: Path = DEFAULT_BASELINE, *, seed_id: str | None = None
    ):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required to run the RFI agent")
        self.settings = settings
        if seed_id is not None and not seed_id.strip():
            raise ValueError("seed_id must be nonempty")
        self.seed_id = seed_id
        self.baseline = Baseline.load(baseline)
        self.corpus = load_corpus()
        manifest_path = settings.data_dir / "corpus" / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Rendered corpus is missing; run 'hopper corpus build' first")
        manifest = json.loads(manifest_path.read_text())
        if manifest["corpus_version"] != self.corpus.version:
            raise ValueError("Rendered corpus is stale; run 'hopper corpus build' first")
        hashes = {entry["id"]: entry["sha256"] for entry in manifest["drawings"]}
        self.images = {}
        for identifier in self.corpus.drawings:
            png = (settings.data_dir / "corpus" / "drawings" / f"{identifier}.png").read_bytes()
            if hashlib.sha256(png).hexdigest() != hashes[identifier]:
                raise ValueError(
                    f"Drawing {identifier} differs from its manifest; rebuild the corpus"
                )
            self.images[identifier] = otel.Image(identifier, png)
        self.ledger = Ledger(settings)
        self.ledger.initialize()

    def run(
        self, task: Task, repeat: int = 0, conversation: Conversation | None = None
    ) -> RunResult:
        if task.corpus_version != self.corpus.version:
            raise ValueError(f"Task {task.id} uses a different corpus version")
        if repeat < 0:
            raise ValueError("repeat must be nonnegative")
        return _Turn(self, task, repeat, conversation or Conversation()).run()

    def batch(
        self, tasks: list[Task], repeats: int, *, followups: bool = True, progress=None
    ) -> dict:
        if repeats < 1 or not tasks:
            raise ValueError("Batch needs tasks and at least one repeat")
        started = perf_counter()
        results = []
        cancelled = Event()

        def conversation_runs(task, repeat):
            conversation = Conversation(cancelled=cancelled)
            turns = []
            questions = [
                task.question,
                *FOLLOWUP_QUESTIONS[: followup_count(task.id, repeat) if followups else 0],
            ]
            for question in questions:
                if cancelled.is_set():
                    break
                result = self.run(
                    Task(task.id, question, task.corpus_version), repeat, conversation
                )
                turns.append(result)
                if result.should_stop_batch:
                    cancelled.set()
                if result.error or result.answer is None:
                    break
            return turns

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            try:
                for repeat in range(repeats):
                    for task in tasks:
                        futures.append(pool.submit(conversation_runs, task, repeat))
                for future in as_completed(futures):
                    for result in future.result():
                        results.append(result)
                        if progress:
                            progress(result)
                        if result.should_stop_batch:
                            raise AgentError(f"Batch stopped after {result.error}")
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        base = [result for result in results if result.turn == 0]
        follow = [result for result in results if result.turn > 0]
        missing = sum(result.answer is None for result in results)
        return {
            "base_runs": len(base),
            "followup_runs": len(follow),
            "runs": len(results),
            "answered": len(results) - missing,
            "without_answer": missing,
            "base_without_answer": sum(result.answer is None for result in base),
            "followup_without_answer": sum(result.answer is None for result in follow),
            "without_answer_rate": missing / len(results),
            "errors": sum(result.error is not None for result in results),
            "export_failures": sum(
                result.answer is not None and result.error is not None for result in results
            ),
            "settled_usd": str(
                sum((Decimal(result.settled_usd) for result in results), Decimal(0))
            ),
            "held_usd": str(sum((Decimal(result.held_usd) for result in results), Decimal(0))),
            "spend_cap_usd": str(self.settings.spend_cap_usd),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "results": [
                asdict(result)
                for result in sorted(results, key=lambda r: (r.repeat, r.task_id, r.turn))
            ],
        }


class _Turn:
    def __init__(self, agent: Agent, task: Task, repeat: int, conversation: Conversation):
        self.agent, self.task, self.repeat, self.conversation = agent, task, repeat, conversation
        self.prefix = (
            f"seed:{agent.seed_id}:{uuid4().hex}:" if agent.seed_id else f"agent:{uuid4().hex}:"
        )
        self.telemetry = otel.Telemetry(
            agent.settings,
            task.id,
            agent.baseline.version,
            resource_attributes={"hopper.seed.id": agent.seed_id} if agent.seed_id else {},
        )
        self.read_drawings: set[str] = set()

    def chat(self, stage: str, messages: list[dict], tools: list[dict]) -> dict:
        if self.conversation.cancelled.is_set():
            raise AgentError("Batch cancelled before the next model request")
        baseline = self.agent.baseline
        system = [
            {
                "type": "text",
                "text": baseline.prompts["shared"],
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": baseline.prompts[stage]},
        ]
        attributes = {
            "gen_ai.input.messages": otel.json_content(otel.trace_messages(messages)),
            "gen_ai.system_instructions": otel.json_content(
                otel.trace_messages([{"role": "system", "content": system}])[0]["parts"]
            ),
            "gen_ai.tool.definitions": otel.json_content(
                [
                    {
                        "type": "function",
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    }
                    for tool in tools
                ]
            ),
            "gen_ai.request.max_tokens": MAX_TOKENS,
        }
        with self.telemetry.span(
            "chat", MODEL, conversation_id=self.conversation.id, attributes=attributes
        ) as span:
            payload = {
                "model": MODEL,
                "system": system,
                "messages": otel.wire_content(messages),
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": baseline.effort},
            }
            if stage == "triage":
                payload["output_config"]["format"] = {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rfi_type": {"type": "string"},
                            "mentioned_ids": {"type": "array", "items": {"type": "string"}},
                            "needs_calculation": {"type": "boolean"},
                        },
                        "required": ["rfi_type", "mentioned_ids", "needs_calculation"],
                        "additionalProperties": False,
                    },
                }
            if tools:
                payload.update(
                    tools=tools, tool_choice={"type": "auto", "disable_parallel_tool_use": True}
                )
            count = _count_tokens(self.agent.settings, payload)
            if type(count) is not int or not 0 <= count <= 98_976:
                raise AgentError("Input token count is invalid or exceeds the 100000-token budget")
            payload["max_tokens"] = MAX_TOKENS
            if self.conversation.cancelled.is_set():
                raise AgentError("Batch cancelled before the next model request")

            def request():
                if self.conversation.cancelled.is_set():
                    return CallResult(None, TokenUsage(0, 0))
                response = _post_json(self.agent.settings, "messages", payload)
                return CallResult(response, _usage(response))

            response = self.agent.ledger.call(
                self.prefix + stage, TokenBudget(MODEL, count + 1024, MAX_TOKENS), request
            )
            if response is None:
                raise AgentError("Batch cancelled before the next model request")
            usage = _usage(response)
            span.set_attributes(
                {
                    "gen_ai.response.id": response["id"],
                    "gen_ai.response.model": response["model"],
                    "gen_ai.response.finish_reasons": [response["stop_reason"]],
                    "gen_ai.usage.input_tokens": usage.total_input_tokens,
                    "gen_ai.usage.output_tokens": usage.output_tokens,
                    "gen_ai.usage.cache_read.input_tokens": usage.cache_read_tokens,
                    "gen_ai.usage.cache_creation.input_tokens": usage.cache_write_tokens,
                    "gen_ai.output.messages": otel.json_content(
                        otel.trace_messages([{"role": "assistant", "content": response["content"]}])
                    ),
                }
            )
            if response["model"] != MODEL:
                raise AgentError(f"Provider returned unexpected model {response['model']}")
            if response["stop_reason"] in {"refusal", "max_tokens"}:
                raise ModelStopped(response["stop_reason"])
            if response["stop_reason"] not in {"end_turn", "tool_use"}:
                raise ModelStopped(f"Unexpected stop reason {response['stop_reason']}")
            return response

    def tool(self, call: dict) -> tuple[dict, dict | None]:
        name, arguments = call["name"], call["input"]
        terminal = None
        result = {"type": "tool_result", "tool_use_id": call["id"]}
        with self.telemetry.span(
            "execute_tool",
            name,
            conversation_id=self.conversation.id,
            attributes={
                "gen_ai.tool.call.id": call["id"],
                "gen_ai.tool.call.arguments": otel.json_content(arguments),
            },
        ) as span:
            self.conversation.tool_number += 1
            try:
                _arguments(name, arguments, self.agent.corpus)
                if name == "report_findings":
                    cited = {citation.split("/")[0] for citation in arguments["citations"]}
                    unread = (cited & self.agent.images.keys()) - self.read_drawings
                    if unread:
                        raise ValueError(
                            f"Use read_drawing for {', '.join(sorted(unread))} "
                            "before reporting findings"
                        )
                if name not in {"report_findings", "submit_answer"} and tool_fault(
                    self.task.id, self.repeat, self.conversation.tool_number
                ):
                    fault = (ToolUnavailable, ToolTimedOut, ToolRateLimited)[
                        self.conversation.tool_number % 3
                    ]
                    raise fault(
                        "Simulated transient document service failure; retry this tool call"
                    )
                content = self.execute(name, arguments)
                if name in {"report_findings", "submit_answer"}:
                    terminal = arguments
                result["content"] = content
            except (ValueError, RuntimeError, OSError) as error:
                otel.record_error(span, error)
                result.update(
                    is_error=True,
                    content=[{"type": "text", "text": f"{type(error).__name__}: {error}"}],
                )
            span.set_attribute("gen_ai.tool.call.result", otel.json_content(result["content"]))
            span.set_attribute(
                "gen_ai.output.messages",
                otel.json_content(otel.trace_messages([{"role": "user", "content": [result]}])),
            )
        return result, terminal

    def execute(self, name: str, args: dict) -> list:
        corpus = self.agent.corpus
        if name == "search_documents":
            text = json.dumps(search_corpus(corpus, args["query"], 6), ensure_ascii=False)
        elif name == "read_section":
            identifier = args["id"]
            if identifier in corpus.sections:
                section = corpus.sections[identifier]
                text = f"## {section.id} {section.title}\n\n{section.text}"
            elif identifier in corpus.documents:
                text = corpus.documents[identifier]
            else:
                raise ValueError(f"Unknown section or document: {identifier}")
        elif name == "list_addenda":
            text = "\n\n".join(corpus.documents[identifier] for identifier in ADDENDUM_IDS)
        elif name == "read_drawing":
            if args["id"] not in self.agent.images:
                raise ValueError(f"Unknown drawing: {args['id']}")
            self.read_drawings.add(args["id"])
            return [
                {
                    "type": "text",
                    "text": f"Drawing {args['id']}. Cite numbered notes on the sheet.",
                },
                self.agent.images[args["id"]],
            ]
        elif name == "calculate":
            text = calculate(args["code"])
        else:
            text = json.dumps(args, ensure_ascii=False)
        return [{"type": "text", "text": text}]

    def research(self, messages: list[dict]) -> dict:
        tools = self.agent.baseline.tools([name for name in TOOL_FIELDS if name != "submit_answer"])
        with self.telemetry.span(
            "invoke_agent",
            "research",
            conversation_id=self.conversation.id,
            attributes={"gen_ai.input.messages": otel.json_content(otel.trace_messages(messages))},
        ) as span:
            for step in range(self.agent.baseline.research_step_cap):
                response = self.chat("research", messages, tools)
                messages.append({"role": "assistant", "content": response["content"]})
                calls = [block for block in response["content"] if block["type"] == "tool_use"]
                if len(calls) != 1 or calls[0]["name"] == "submit_answer":
                    raise AgentError("Research must return exactly one research tool call")
                result, findings = self.tool(calls[0])
                if findings is not None:
                    span.set_attribute(
                        "gen_ai.output.messages",
                        otel.json_content(
                            otel.trace_messages(
                                [otel.text_message("assistant", json.dumps(findings))]
                            )
                        ),
                    )
                    return findings
                remaining = self.agent.baseline.research_step_cap - step - 1
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            result,
                            {
                                "type": "text",
                                "text": f"{remaining} calls remain, including report_findings.",
                            },
                        ],
                    }
                )
            raise AgentError("Research step cap reached without report_findings")

    def run(self) -> RunResult:
        started = perf_counter()
        answer, error, error_type, trace_id = None, None, None, ""
        turn = self.conversation.turn
        messages = [*self.conversation.history, otel.text_message("user", self.task.question)]
        try:
            with self.telemetry:
                with self.telemetry.span(
                    "invoke_agent",
                    "rfi",
                    conversation_id=self.conversation.id,
                    root=True,
                    attributes={
                        "gen_ai.input.messages": otel.json_content(otel.trace_messages(messages)),
                        "hopper.repeat": self.repeat,
                        "hopper.turn": turn,
                        "hopper.research.step_cap": self.agent.baseline.research_step_cap,
                    },
                ) as span:
                    trace_id = format(span.get_span_context().trace_id, "032x")
                    response = self.chat("triage", messages, [])
                    triage = json.loads(
                        "".join(
                            block["text"]
                            for block in response["content"]
                            if block["type"] == "text"
                        )
                    )
                    if not isinstance(triage, dict) or triage.keys() != {
                        "rfi_type",
                        "mentioned_ids",
                        "needs_calculation",
                    }:
                        raise AgentError(
                            "Triage must return rfi_type, mentioned_ids, needs_calculation"
                        )
                    if (
                        not isinstance(triage["rfi_type"], str)
                        or type(triage["needs_calculation"]) is not bool
                        or not isinstance(triage["mentioned_ids"], list)
                        or any(not isinstance(value, str) for value in triage["mentioned_ids"])
                    ):
                        raise AgentError("Triage fields have invalid types")
                    findings = self.research(
                        [*messages, otel.text_message("user", "Triage: " + json.dumps(triage))]
                    )
                    response = self.chat(
                        "writer",
                        [
                            *messages,
                            otel.text_message("user", "Research findings: " + json.dumps(findings)),
                        ],
                        self.agent.baseline.tools(["submit_answer"]),
                    )
                    calls = [block for block in response["content"] if block["type"] == "tool_use"]
                    if len(calls) != 1 or calls[0]["name"] != "submit_answer":
                        raise AgentError("Writer must return exactly one submit_answer call")
                    _, answer = self.tool(calls[0])
                    if answer is None:
                        raise AgentError("submit_answer failed")
                    span.set_attribute(
                        "gen_ai.output.messages",
                        otel.json_content(
                            otel.trace_messages(
                                [
                                    otel.text_message(
                                        "assistant", json.dumps(answer, ensure_ascii=False)
                                    )
                                ]
                            )
                        ),
                    )
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as failure:
            error_type = type(failure).__name__
            error = f"{type(failure).__name__}: {failure}"
            if notes := getattr(failure, "__notes__", None):
                error += "; " + "; ".join(notes)
        self.conversation.turn += 1
        if answer is not None:
            self.conversation.history.extend(
                [
                    otel.text_message("user", self.task.question),
                    otel.text_message("assistant", json.dumps(answer)),
                ]
            )
        costs = [
            total for total in self.agent.ledger.summary() if total.stage.startswith(self.prefix)
        ]
        return RunResult(
            self.task.id,
            self.repeat,
            turn,
            self.conversation.id,
            trace_id,
            answer,
            error,
            str(sum((cost.settled_usd for cost in costs), Decimal(0))),
            str(sum((cost.held_usd for cost in costs), Decimal(0))),
            round(perf_counter() - started, 3),
            error_type,
        )
