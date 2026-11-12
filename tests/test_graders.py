import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import Mock

import pytest
import yaml

from hopper.graders import (
    Catalog,
    CodeGrader,
    JevGrader,
    ThresholdValidation,
    Validation,
    definition_fingerprint,
    read_definition,
    threshold_pairs,
)
from hopper.query import connect


@pytest.fixture
def catalog(tmp_path):
    (tmp_path / "graders").mkdir()
    return Catalog(tmp_path)


@pytest.fixture
def definition():
    return {
        "name": "grounded",
        "kind": "jev",
        "applies_to": "rfi-agent",
        "question": "Every claim is supported by the retrieved text.",
        "fields": ["final_answer", "retrieved_text"],
        "yes_threshold": 0.7,
        "no_threshold": 0.3,
        "model": "jev-1.14.0",
        "origin": "human",
    }


def write_validation(catalog, grader, validation):
    path = catalog.directory / grader.name / f"v{grader.version}.yaml"
    document = yaml.safe_load(path.read_text())
    document["validation"] = validation
    path.write_text(yaml.safe_dump(document))


def test_seed_installation_is_active_pinned_and_preserves_existing_versions(catalog):
    resolver = Mock(return_value="jev-1.14.0")
    installed = catalog.install_seeds(resolver)
    resolver.assert_called_once_with()
    assert {grader.name for grader in installed} == {
        "answered_question",
        "grounded",
        "actionable",
        "cites_sources",
    }
    assert all(grader.status == "active" and grader.version == 1 for grader in installed)
    assert all(grader.applies_to == "rfi-agent" and grader.origin == "seed" for grader in installed)
    assert {grader.model for grader in installed if isinstance(grader, JevGrader)} == {"jev-1.14.0"}
    code = next(grader for grader in installed if isinstance(grader, CodeGrader))
    assert code.function == "hopper.graders.code:cites_sources"
    assert code.fields == ("final_answer",)
    assert code.validation is None
    assert read_definition(catalog.directory / "cites_sources" / "v1.yaml") == code
    grounded = next(grader for grader in installed if grader.name == "grounded")
    write_validation(catalog, grounded, {"labels": 50, "agreement": 0.9})

    catalog.retire("grounded", 1)
    saved = {path: path.read_bytes() for path in catalog.directory.glob("*/v*.yaml")}
    second_resolver = Mock(side_effect=AssertionError("Must reuse installed graders"))
    Catalog(catalog.directory.parent).install_seeds(second_resolver)
    second_resolver.assert_not_called()
    assert {path: path.read_bytes() for path in saved} == saved
    assert len(catalog.list_graders()) == 4


@pytest.mark.parametrize("model", ["jev-latest", "jev-preview", "", "other-model"])
def test_seed_installation_rejects_unresolved_model_before_writing(catalog, model):
    with pytest.raises(ValueError, match="model"):
        catalog.install_seeds(Mock(return_value=model))
    assert catalog.list_graders() == []


def test_interrupted_seed_installation_reuses_durable_pin(catalog, monkeypatch):
    replace = os.replace

    def fail_second_grader(source, destination):
        if destination.parent.name == "grounded":
            raise OSError("interrupted publication")
        replace(source, destination)

    resolver = Mock(return_value="jev-1.14.0")
    with monkeypatch.context() as patch:
        patch.setattr("hopper.graders.os.replace", fail_second_grader)
        with pytest.raises(OSError, match="interrupted publication"):
            catalog.install_seeds(resolver)
    assert [grader.name for grader in catalog.list_graders()] == ["answered_question"]

    restarted_resolver = Mock(side_effect=AssertionError("Must reuse the saved model pin"))
    installed = Catalog(catalog.directory.parent).install_seeds(restarted_resolver)
    assert len(installed) == 4
    assert {grader.model for grader in installed if isinstance(grader, JevGrader)} == {"jev-1.14.0"}
    resolver.assert_called_once_with()
    restarted_resolver.assert_not_called()


def test_new_versions_are_drafts_and_do_not_rewrite_predecessors(catalog, definition):
    catalog.install_seeds(Mock(return_value="jev-1.14.0"))
    first_path = catalog.directory / "grounded" / "v1.yaml"
    original = first_path.read_bytes()
    definition["model"] = "jev-1.15.0"
    second = catalog.create_draft(definition)
    assert second.version == 2
    assert second.status == "draft"
    assert second.validation is None
    assert second.model == "jev-1.15.0"
    assert first_path.read_bytes() == original
    assert read_definition(catalog.directory / "grounded" / "v2.yaml") == second


def test_concurrent_creation_allocates_distinct_versions(catalog, definition):
    def create(_):
        return Catalog(catalog.directory.parent).create_draft(definition)

    with ThreadPoolExecutor(max_workers=8) as workers:
        graders = list(workers.map(create, range(8)))
    assert sorted(grader.version for grader in graders) == list(range(1, 9))
    assert len(catalog.list_graders()) == 8


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", "../escape"),
        ("model", "jev-latest"),
        ("fields", ["unknown"]),
        ("fields", ["final_answer", "final_answer"]),
        ("fields", []),
        ("yes_threshold", float("nan")),
        ("yes_threshold", 0.3),
        ("no_threshold", True),
        ("status", "active"),
        ("version", 99),
        ("validation", {"labels": 50, "agreement": 1}),
        ("unknown", "value"),
    ],
)
def test_invalid_draft_is_rejected_without_a_version_file(catalog, definition, key, value):
    definition[key] = value
    with pytest.raises(ValueError):
        catalog.create_draft(definition)
    assert catalog.list_graders() == []


def test_read_rejects_definition_that_disagrees_with_its_path(catalog, definition):
    grader = catalog.create_draft(definition)
    path = catalog.directory / grader.name / "v1.yaml"
    document = yaml.safe_load(path.read_text())
    document["version"] = 2
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="file path"):
        catalog.list_graders()


@pytest.mark.parametrize(
    "validation",
    [None, {"labels": 49, "agreement": 1}, {"labels": 50, "agreement": 0.849}],
)
def test_promotion_requires_label_count_and_agreement(catalog, definition, validation):
    grader = catalog.create_draft(definition)
    write_validation(catalog, grader, validation)
    runner = Mock()
    with pytest.raises(ValueError, match="50 labels and 0.85"):
        catalog.promote(grader.name, grader.version, run_code_tests=runner)
    assert catalog.list_graders()[0].status == "draft"
    runner.assert_not_called()


def test_promotion_keeps_predecessor_active_and_pin_unchanged(catalog, definition):
    catalog.install_seeds(Mock(return_value="jev-1.14.0"))
    grader = catalog.create_draft(definition)
    report = Validation(
        labels=50,
        agreement=0.85,
        threshold_sweep=tuple(
            ThresholdValidation(no, yes, 0.85, None, None) for no, yes in threshold_pairs(grader)
        ),
        validated_at="2026-11-11T00:00:00+00:00",
        definition_hash=definition_fingerprint(grader),
        labels_hash="a" * 64,
    )
    write_validation(catalog, grader, asdict(report))
    runner = Mock()
    promoted = catalog.promote(grader.name, grader.version, run_code_tests=runner)
    assert promoted.status == "active"
    assert promoted.model == grader.model
    assert [item.status for item in catalog.list_graders() if item.name == "grounded"] == [
        "active",
        "active",
    ]
    runner.assert_not_called()
    with pytest.raises(ValueError, match="Only draft"):
        catalog.promote(grader.name, grader.version, run_code_tests=runner)
    assert catalog.retire(grader.name, grader.version).status == "retired"
    with pytest.raises(ValueError, match="Only draft"):
        catalog.promote(grader.name, grader.version, run_code_tests=runner)
    with pytest.raises(ValueError, match="cannot transition"):
        catalog.retire(grader.name, grader.version)


def test_draft_can_be_retired_without_promotion(catalog, definition):
    grader = catalog.create_draft(definition)
    assert catalog.retire(grader.name, grader.version).status == "retired"


def test_code_promotion_runs_tests_and_failure_leaves_draft(catalog):
    grader = catalog.create_draft(
        {
            "name": "cites_sources",
            "kind": "code",
            "applies_to": "*",
            "fields": ["final_answer"],
            "origin": "human",
            "function": "hopper.graders.code:cites_sources",
        }
    )
    failing = Mock(side_effect=RuntimeError("code tests failed"))
    with pytest.raises(RuntimeError, match="code tests failed"):
        catalog.promote(grader.name, grader.version, run_code_tests=failing)
    failing.assert_called_once_with(grader)
    assert catalog.list_graders()[0].status == "draft"
    passing = Mock()
    promoted = catalog.promote(grader.name, grader.version, run_code_tests=passing)
    passing.assert_called_once_with(grader)
    assert isinstance(promoted, CodeGrader)
    assert promoted.status == "active"


@pytest.mark.parametrize("is_code", [False, True])
def test_conflicting_seed_is_rejected_before_installing_anything(catalog, definition, is_code):
    if is_code:
        definition = {
            "name": "grounded",
            "kind": "code",
            "applies_to": "rfi-agent",
            "fields": ["final_answer"],
            "origin": "seed",
            "function": "hopper.graders.code:cites_sources",
        }
    conflicting = catalog.create_draft(definition)
    resolver = Mock(return_value="jev-1.14.0")
    with pytest.raises(ValueError, match="conflicts with its seed definition"):
        catalog.install_seeds(resolver)
    resolver.assert_not_called()
    assert catalog.list_graders() == [conflicting]


@pytest.mark.parametrize(
    "extra",
    [
        "model: jev-1.15.0\n",
        "status: active\n",
        "validation:\n  labels: 50\n  labels: 1\n  agreement: 1\n",
    ],
)
def test_duplicate_yaml_keys_are_rejected(catalog, definition, extra):
    grader = catalog.create_draft(definition)
    path = catalog.directory / grader.name / "v1.yaml"
    document = path.read_text().replace("validation: null\n", "")
    path.write_text(document + extra)
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        read_definition(path)


@pytest.mark.parametrize("name", ["trace_id", "service", "finished"])
def test_promotion_rejects_trace_column_names_and_keeps_queries_working(catalog, definition, name):
    definition["name"] = name
    grader = catalog.create_draft(definition)
    write_validation(catalog, grader, {"labels": 50, "agreement": 1})
    path = catalog.directory / name / "v1.yaml"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="collide"):
        catalog.promote(name, 1, run_code_tests=Mock())
    assert path.read_bytes() == before
    with connect(catalog.directory.parent) as connection:
        assert connection.execute("SELECT count(*) FROM traces").fetchone() == (0,)
