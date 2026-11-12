import pytest

from hopper.corpus import load_corpus
from hopper.graders.code import cites_sources


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_cites_sources_accepts_document_section_drawing_and_addendum_ids(corpus):
    citations = [
        "SPEC-01",
        "SPEC-01/1.1",
        "M-101",
        corpus.drawings["M-101"].notes[0].id,
        "ADD-1/1",
        "SPEC-01",
    ]
    result = cites_sources({"final_answer": {"citations": citations}}, corpus)
    assert result.passed
    assert result.reason is None


def test_cites_sources_rejects_unknown_section_even_when_document_exists(corpus):
    result = cites_sources({"final_answer": {"citations": ["SPEC-01", "SPEC-01/999"]}}, corpus)
    assert not result.passed
    assert "SPEC-01/999" in result.reason


@pytest.mark.parametrize(
    "answer", [None, {}, {"citations": []}, {"citations": "SPEC-01"}, {"citations": [1]}]
)
def test_cites_sources_requires_a_nonempty_list_of_ids(corpus, answer):
    result = cites_sources({"final_answer": answer}, corpus)
    assert not result.passed
    assert result.reason


def test_cites_sources_does_not_treat_incidental_prose_as_citations(corpus):
    result = cites_sources({"final_answer": {"text": "See SPEC-01.", "citations": []}}, corpus)
    assert not result.passed
    assert result.reason
