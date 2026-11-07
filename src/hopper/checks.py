import re
from decimal import Decimal, InvalidOperation

MODES = (
    "lookup",
    "multi_section",
    "missed_addendum",
    "conflict",
    "needs_engineer",
    "calculation",
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _number(value) -> Decimal | None:
    if type(value) not in (int, float):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def check_answer(key: dict, answer: dict) -> dict:
    """Evaluate only the section 4.1 criteria applicable to this answer key."""
    key = key if isinstance(key, dict) else {}
    answer = answer if isinstance(answer, dict) else {}
    mode = key.get("mode")
    text = answer.get("text")
    citations = answer.get("citations")
    has_citations = isinstance(citations, list) and all(
        isinstance(citation, str) for citation in citations
    )
    cited = set(citations) if has_citations else set()
    criteria = {}
    if mode != "needs_engineer":
        required = key.get("required_citations")
        criteria["citations"] = (
            has_citations
            and isinstance(required, list)
            and all(isinstance(citation, str) and citation in cited for citation in required)
        )
    phrases = key.get("required_phrases")
    if phrases:
        criteria["phrases"] = (
            isinstance(text, str)
            and isinstance(phrases, list)
            and all(
                isinstance(phrase, str) and _normalize(phrase) in _normalize(text)
                for phrase in phrases
            )
        )
    if mode == "missed_addendum":
        citation = key.get("addendum_citation")
        criteria["addendum_cited"] = isinstance(citation, str) and citation in cited
    if mode == "conflict":
        documents = key.get("conflict_documents")
        criteria["both_sides_named"] = (
            isinstance(text, str)
            and isinstance(documents, list)
            and len(documents) == 2
            and all(
                isinstance(document, str)
                and re.search(rf"(?<![\w-]){re.escape(document)}(?![\w-])", text) is not None
                for document in documents
            )
        )
        citation = key.get("precedence_citation")
        criteria["precedence_cited"] = isinstance(citation, str) and citation in cited
    criteria["escalation"] = (
        mode in MODES
        and type(answer.get("needs_engineer")) is bool
        and type(key.get("needs_engineer")) is bool
        and answer["needs_engineer"] == key["needs_engineer"]
    )
    if mode == "calculation":
        value = _number(answer.get("value"))
        expected = _number(key.get("value"))
        tolerance = _number(key.get("tolerance"))
        criteria["value_in_tolerance"] = (
            value is not None
            and expected is not None
            and tolerance is not None
            and tolerance >= 0
            and abs(value - expected) <= tolerance
        )
    return {"pass": all(criteria.values()), "criteria": criteria}
