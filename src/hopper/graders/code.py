from dataclasses import dataclass

from hopper.corpus import Corpus


@dataclass(frozen=True)
class CodeResult:
    passed: bool
    reason: str | None


def cites_sources(state: dict, corpus: Corpus) -> CodeResult:
    answer = state.get("final_answer")
    if not isinstance(answer, dict):
        return CodeResult(False, "No final answer was recorded.")
    citations = answer.get("citations")
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        return CodeResult(False, "Citations must be a list of document or section IDs.")
    if not citations:
        return CodeResult(False, "Answer cites no sources.")
    unknown = sorted(
        {item for item in citations if item not in corpus.documents and item not in corpus.sections}
    )
    if unknown:
        return CodeResult(
            False, "Unknown citation IDs: " + ", ".join(repr(item) for item in unknown)
        )
    return CodeResult(True, None)
