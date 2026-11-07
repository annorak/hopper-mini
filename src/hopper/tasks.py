import json
import math
import operator
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from hopper.checks import MODES, check_answer

if TYPE_CHECKING:
    from hopper.corpus import Corpus

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "rfis.jsonl"
TASK_FIELDS = {
    "id", "mode", "question", "corpus_version", "case_id", "answer_key", "gold_answer",
    "generation",
}
KEY_FIELDS = {"mode", "required_citations", "required_phrases", "needs_engineer"}
MODE_FIELDS = {
    "lookup": set(),
    "multi_section": set(),
    "missed_addendum": {"addendum_citation"},
    "conflict": {"conflict_documents", "precedence_citation"},
    "needs_engineer": set(),
    "calculation": {"value", "tolerance"},
}
NUMBER_TOKEN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")


def _phrase(text: str) -> str:
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0].split())


def _is_number(value) -> bool:
    return type(value) is int or type(value) is float and math.isfinite(value)


def _quote(corpus: "Corpus", citation: str, quote: str) -> str:
    if not quote or quote not in corpus.sections[citation].text:
        raise ValueError(f"Source quote does not match {citation}: {quote!r}")
    return quote


def make_plans(corpus: "Corpus") -> list[dict]:
    """Build the fixed RFI schedule and keys from the authoritative source corpus."""
    plans = []

    def add(mode, case_id, brief, citations, phrases, **fields):
        key = {
            "mode": mode,
            "required_citations": [] if mode == "needs_engineer" else citations,
            "required_phrases": phrases,
            "needs_engineer": mode == "needs_engineer",
            **fields,
        }
        sources = {
            citation: f"{corpus.sections[citation].title}\n{corpus.sections[citation].text}"
            for citation in citations
        }
        plans.append({
            "id": f"RFI-{len(plans) + 1:03d}",
            "mode": mode,
            "case_id": case_id,
            "corpus_version": corpus.version,
            "answer_key": key,
            "brief": brief,
            "sources": sources,
        })

    blocked = {case["section"] for case in corpus.cases["changes"] + corpus.cases["gaps"]}
    blocked.update(case["spec"] for case in corpus.cases["conflicts"])
    safe = sorted(
        (section for section in corpus.sections.values()
         if section.kind == "spec" and section.id not in blocked),
        key=lambda section: section.id,
    )
    groups = {}
    for section in safe:
        groups.setdefault(section.document_id, []).append(section)
    for sections in groups.values():
        sections.sort(key=lambda section: (section.id.endswith("/1.1"), section.id))
    documents = sorted(groups)
    if len(documents) != 20 or any(len(sections) < 3 for sections in groups.values()):
        raise ValueError("Task plans require 20 specs with three unaffected sections each")
    for index in range(25):
        document = documents[index % len(documents)]
        section = groups[document][index // len(documents)]
        add("lookup", section.id,
            f"Ask a focused contractor RFI about {section.title} under {section.id}. "
            "The answer must state the documented requirement and cite its section.",
            [section.id], [_phrase(section.text)])
    for index in range(25):
        document = documents[index % len(documents)]
        offset = index // len(documents)
        sections = groups[document][offset:offset + 2]
        citations = [section.id for section in sections]
        add("multi_section", "+".join(citations),
            f"Ask one RFI reviewing {sections[0].title} and {sections[1].title} in {document}. "
            "Request both documented requirements and a citation for each, without inventing a "
            "technical dependency between them.",
            citations, [_phrase(section.text) for section in sections])

    change_scenarios = (
        "a material submittal prepared from the original specification",
        "a purchase order being checked before release",
        "a field installation instruction that still quotes the base requirement",
        "a bid clarification comparing the base specification with the issued addendum",
        "an inspection checklist that must reflect the current contract",
    )
    changes = corpus.cases["changes"]
    for index in range(25):
        case = changes[index % len(changes)]
        _quote(corpus, case["section"], case["before"])
        after = _quote(corpus, case["id"], case["after"])
        scenario = change_scenarios[index // len(changes)]
        add("missed_addendum", case["id"],
            f"Draft an RFI about {case['topic']} for {scenario}. Ask what requirement now "
            f"governs {case['section']} and which addendum item changed it. Cite both sources.",
            [case["section"], case["id"]], [after], addendum_citation=case["id"])

    conflict_scenarios = (
        "procurement needs a selection before ordering",
        "the shop drawing reviewer has found inconsistent instructions",
        "the installer needs a direction before starting work",
        "the inspector has held the work pending contract clarification",
        "the coordination team must correct its installation schedule",
    )
    conflicts = corpus.cases["conflicts"]
    precedence = corpus.cases["precedence"]
    for index in range(25):
        case = conflicts[index % len(conflicts)]
        citations = [case["spec"], case["drawing"], precedence]
        _quote(corpus, case["spec"], case["spec_quote"])
        _quote(corpus, case["drawing"], case["drawing_quote"])
        resolution = case["resolution_quote"]
        if not any(resolution in corpus.sections[citation].text for citation in citations):
            raise ValueError(f"Conflict resolution quote does not match {case['id']}")
        documents = [corpus.sections[citation].document_id for citation in citations[:2]]
        add("conflict", case["id"],
            f"Draft an RFI about the {case['topic']} conflict because "
            f"{conflict_scenarios[index // len(conflicts)]}. Request the conflicting documents "
            "by name, their governing requirement, and the order-of-precedence citation.",
            citations, [resolution], conflict_documents=documents,
            precedence_citation=precedence)

    gap_scenarios = (
        "the contractor needs a design decision before preparing its work plan",
        "procurement needs the unresolved design criterion before selecting equipment",
        "the field team needs an engineer's direction before releasing the affected work",
        "the estimator needs a design decision before finalizing the affected scope",
        "the submittal reviewer needs the engineer to supply the missing design information",
    )
    gaps = corpus.cases["gaps"]
    for index in range(25):
        case = gaps[index % len(gaps)]
        quote = _quote(corpus, case["section"], case["quote"])
        add("needs_engineer", case["id"],
            f"Draft an RFI about {case['topic']} because {gap_scenarios[index // len(gaps)]}. "
            "Identify the documented gap and request an engineer-of-record decision. "
            "Do not invent the missing design value.", [case["section"]], [quote])

    operations = {"add": operator.add, "subtract": operator.sub,
                  "multiply": operator.mul, "divide": operator.truediv}
    calculations = corpus.cases["calculations"]
    if len(calculations) != 25:
        raise ValueError("Exactly 25 source-backed calculation cases are required")
    for case in calculations:
        values = []
        citations = []
        for item in case["inputs"]:
            quote = _quote(corpus, item["citation"], item["quote"])
            tokens = NUMBER_TOKEN.findall(quote)
            if len(tokens) != 1:
                raise ValueError(f"Calculation input must quote one numeric value: {case['id']}")
            value = Decimal(tokens[0])
            if not _is_number(item["value"]) or Decimal(str(item["value"])) != value:
                raise ValueError(f"Calculation value differs from its source quote: {case['id']}")
            values.append(value)
            citations.append(item["citation"])
        if len(values) != 2 or case["operation"] not in operations:
            raise ValueError(f"Invalid calculation operation or inputs: {case['id']}")
        value = operations[case["operation"]](*values)
        add("calculation", case["id"],
            f"Ask a calculation RFI about {case['topic']}. Apply {case['operation']} to "
            f"{case['inputs'][0]['name']} and {case['inputs'][1]['name']} in that order. "
            f"Request the result in {case['unit']} and cite both input sections.",
            list(dict.fromkeys(citations)), [], value=float(value),
            tolerance=float(case["tolerance"]))
    return plans


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _answer_errors(answer, corpus: "Corpus") -> list[str]:
    if not isinstance(answer, dict):
        return ["gold_answer must be an object"]
    errors = []
    required = {"text", "citations", "needs_engineer"}
    if not required <= answer.keys() or answer.keys() - required - {"value"}:
        errors.append("gold_answer has missing or unexpected fields")
    if not isinstance(answer.get("text"), str) or not answer["text"].strip():
        errors.append("gold_answer.text must be nonempty prose")
    if not _strings(answer.get("citations")):
        errors.append("gold_answer.citations must be a list of nonempty strings")
    else:
        for citation in answer["citations"]:
            if citation not in corpus.sections and citation not in corpus.documents:
                errors.append(f"gold_answer has unknown citation {citation}")
    if type(answer.get("needs_engineer")) is not bool:
        errors.append("gold_answer.needs_engineer must be a boolean")
    if "value" in answer and not _is_number(answer["value"]):
        errors.append("gold_answer.value must be a finite JSON number")
    return errors


def _key_errors(key, corpus: "Corpus") -> list[str]:
    if not isinstance(key, dict):
        return ["answer_key must be an object"]
    mode = key.get("mode")
    if not isinstance(mode, str) or mode not in MODES:
        return ["answer_key.mode is unknown"]
    errors = []
    if key.keys() != KEY_FIELDS | MODE_FIELDS[mode]:
        errors.append("answer_key has missing or unexpected fields")
    if type(key.get("needs_engineer")) is not bool:
        errors.append("answer_key.needs_engineer must be a boolean")
    for field in ("required_citations", "required_phrases"):
        if not _strings(key.get(field)):
            errors.append(f"answer_key.{field} must be a list of nonempty strings")
    references = list(key["required_citations"]) if _strings(key.get("required_citations")) else []
    for field in ("addendum_citation", "precedence_citation"):
        if field in MODE_FIELDS[mode]:
            if not isinstance(key.get(field), str):
                errors.append(f"answer_key.{field} must be a citation string")
            else:
                references.append(key[field])
    for citation in references:
        if citation not in corpus.sections:
            errors.append(f"answer_key has unknown citation {citation}")
    if mode == "conflict":
        documents = key.get("conflict_documents")
        if not _strings(documents) or len(documents) != 2:
            errors.append("answer_key.conflict_documents must name two documents")
        elif any(document not in corpus.documents for document in documents):
            errors.append("answer_key.conflict_documents contains an unknown document")
    if mode == "calculation":
        for field in ("value", "tolerance"):
            if not _is_number(key.get(field)):
                errors.append(f"answer_key.{field} must be a finite JSON number")
        if _is_number(key.get("tolerance")) and key["tolerance"] < 0:
            errors.append("answer_key.tolerance must be nonnegative")
    return errors


def validate_task(task: dict, corpus: "Corpus", plan: dict) -> list[str]:
    """Validate provider/file records against a source-derived canonical plan."""
    if not isinstance(task, dict):
        return ["task must be an object"]
    errors = []
    if task.keys() != TASK_FIELDS:
        errors.append("task has missing or unexpected fields")
    for field in ("id", "mode", "question", "corpus_version", "case_id"):
        if not isinstance(task.get(field), str) or not task[field].strip():
            errors.append(f"{field} must be a nonempty string")
    for field in ("id", "mode", "case_id", "corpus_version", "answer_key"):
        if task.get(field) != plan[field]:
            errors.append(f"{field} does not match the canonical plan")
    if task.get("corpus_version") != corpus.version:
        errors.append("corpus_version does not match the loaded corpus")
    errors.extend(_key_errors(task.get("answer_key"), corpus))
    errors.extend(_answer_errors(task.get("gold_answer"), corpus))
    generation = task.get("generation")
    if not isinstance(generation, dict):
        errors.append("generation must be an object")
    else:
        if generation.get("model") != "claude-sonnet-5-5":
            errors.append("generation.model must be claude-sonnet-5-5")
        for field in ("response_id", "stage"):
            if not isinstance(generation.get(field), str) or not generation[field].strip():
                errors.append(f"generation.{field} must be a nonempty string")
        digest = generation.get("response_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append("generation.response_sha256 must be a lowercase SHA256 digest")
    result = check_answer(task.get("answer_key"), task.get("gold_answer"))
    if not result["pass"]:
        failed = ", ".join(name for name, passed in result["criteria"].items() if not passed)
        errors.append(f"gold_answer failed checker criteria: {failed}")
    return errors


def validate_tasks(tasks: list[dict], corpus: "Corpus") -> dict:
    plans = {plan["id"]: plan for plan in make_plans(corpus)}
    errors = []
    valid = 0
    counts = dict.fromkeys(MODES, 0)
    ids = []
    questions = []
    if not isinstance(tasks, list):
        tasks = []
        errors.append("tasks must be a list")
    for index, task in enumerate(tasks, 1):
        identity = task.get("id") if isinstance(task, dict) else None
        label = identity if isinstance(identity, str) else f"record {index}"
        if isinstance(identity, str):
            ids.append(identity)
        mode = task.get("mode") if isinstance(task, dict) else None
        if isinstance(mode, str) and mode in counts:
            counts[mode] += 1
        question = task.get("question") if isinstance(task, dict) else None
        if isinstance(question, str):
            questions.append(" ".join(question.casefold().split()))
        if not isinstance(identity, str) or identity not in plans:
            errors.append(f"{label}: unknown task ID")
            continue
        task_errors = validate_task(task, corpus, plans[identity])
        valid += not task_errors
        errors.extend(f"{label}: {error}" for error in task_errors)
    if len(tasks) != 150:
        errors.append(f"Expected 150 tasks, found {len(tasks)}")
    if set(ids) != set(plans):
        errors.append("Task IDs do not cover the complete canonical plan")
    if len(ids) != len(set(ids)):
        errors.append("Task IDs must be unique")
    if len(questions) != len(set(questions)):
        errors.append("Questions must be unique after whitespace and case normalization")
    for mode, count in counts.items():
        if count != 25:
            errors.append(f"Expected 25 {mode} tasks, found {count}")
    return {"valid": valid, "total": len(tasks), "errors": errors, "counts": counts,
            "error_count": len(errors), "corpus_version": corpus.version}


def _reject_constant(value):
    raise ValueError(f"Invalid JSON number {value}")


def load_tasks(path: Path = DEFAULT_DATASET) -> list[dict]:
    tasks = []
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                task = json.loads(line, parse_constant=_reject_constant)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if not isinstance(task, dict):
                raise ValueError(f"{path}:{number}: task must be an object")
            tasks.append(task)
    return tasks


def selfcheck_tasks(tasks: list[dict]) -> dict:
    errors = []
    failed_criteria = Counter()
    for index, task in enumerate(tasks, 1):
        task = task if isinstance(task, dict) else {}
        result = check_answer(task.get("answer_key"), task.get("gold_answer"))
        if not result["pass"]:
            failed = [name for name, passed in result["criteria"].items() if not passed]
            errors.append({"id": task.get("id", f"record {index}"), "failed_criteria": failed})
            failed_criteria.update(failed)
    return {"valid": not errors, "passing": len(tasks) - len(errors), "failing": len(errors),
            "errors": errors, "failed_criteria": dict(failed_criteria)}
