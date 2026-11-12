import json
from dataclasses import asdict
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from typer.main import get_command
from typer.testing import CliRunner
from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage

from hopper.cli import app
from hopper.config import DATA_DIRECTORIES, Settings
from hopper.graders import (
    SEED_CODE,
    Catalog,
    ThresholdValidation,
    Validation,
    definition_fingerprint,
    threshold_pairs,
)
from hopper.graders import jev as jev_module
from hopper.ledger import Ledger

runner = CliRunner()


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    sdk = SimpleNamespace(
        system_one=AsyncMock(
            return_value=SystemOneResponse(
                model="jev-1.14.0",
                usage=Usage(input_tokens=12),
                answers={"resolve_model": NoulAnswer(noul=1.0)},
            )
        ),
        models=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(models=[SimpleNamespace(name="jev-latest")])
            )
        ),
        aclose=AsyncMock(),
    )
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(jev_module, "AsyncTypeSafeClient", factory)
    return tmp_path, sdk, factory


def write_draft(path, **changes):
    document = {
        "name": "answer_complete",
        "kind": "jev",
        "applies_to": "rfi-agent",
        "question": "The answer is complete.",
        "fields": ["final_answer"],
        "yes_threshold": 0.7,
        "no_threshold": 0.3,
    }
    document.update(changes)
    path.write_text(yaml.safe_dump(document))
    return document


def test_help_lists_all_commands():
    expected = set(
        (
            "init spend stack ingest wal writer verify sql trace find corpus tasks agent profile "
            "check replay graders grade label bench chaos evals climb report"
        ).split()
    )
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert set(get_command(app).commands) == expected
    assert all(name in result.output for name in expected)
    for group, commands in (
        ("graders", ("list", "new", "promote", "retire")),
        ("grade", ("work",)),
    ):
        help_result = runner.invoke(app, [group, "--help"])
        assert help_result.exit_code == 0
        assert all(command in help_result.output for command in commands)


def test_init_preserves_existing_files(environment, monkeypatch):
    tmp_path, sdk, factory = environment
    assert runner.invoke(app, ["init"]).exit_code == 0
    assert all((tmp_path / directory).is_dir() for directory in DATA_DIRECTORIES)
    ledger = tmp_path / "ledger.jsonl"
    row = json.loads(ledger.read_text())
    assert row["state"] == "settled" and row["tokens"] == {"input_tokens": 12}
    graders = Catalog(tmp_path).list_graders()
    assert len(graders) == 4 and all(grader.status == "active" for grader in graders)
    saved = {path: path.read_bytes() for path in (tmp_path / "graders").glob("*/v*.yaml")}
    ledger.write_text("existing ledger\n")
    span = tmp_path / "spans" / "existing.parquet"
    span.write_bytes(b"existing data")
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    factory.side_effect = AssertionError("Repeated init must not construct a Jev client")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "10 data directories ready" in result.output
    assert ledger.read_text() == "existing ledger\n"
    assert span.read_bytes() == b"existing data"
    assert {path: path.read_bytes() for path in saved} == saved
    sdk.models.list.assert_awaited_once()
    sdk.system_one.assert_awaited_once()
    sdk.aclose.assert_awaited_once()


def test_first_init_without_key_reports_error_without_grader_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY" in result.output
    assert not list((tmp_path / "graders").glob("*/v*.yaml"))


@pytest.mark.parametrize("model", [None, "jev-latest", "jev-1.14.0"])
def test_new_pins_alias_or_preserves_explicit_model(environment, model):
    data_dir, sdk, factory = environment
    Ledger(Settings(data_dir)).initialize()
    definition = data_dir / "draft.yaml"
    changes = {} if model is None else {"model": model}
    write_draft(definition, **changes)
    result = runner.invoke(app, ["graders", "new", str(definition)])
    assert result.exit_code == 0, result.output
    created = json.loads(result.output)
    assert created["model"] == "jev-1.14.0"
    assert created["version"] == 1 and created["status"] == "draft"
    assert created["origin"] == "human"
    if model == "jev-1.14.0":
        factory.assert_not_called()
    else:
        sdk.system_one.assert_awaited_once()
        sdk.aclose.assert_awaited_once()
    listed = runner.invoke(app, ["graders", "list"])
    assert listed.exit_code == 0
    assert "answer_complete\t1\tdraft\tjev\tjev-1.14.0" in listed.output


def test_new_rejects_duplicate_yaml_keys(environment):
    data_dir, _, factory = environment
    Ledger(Settings(data_dir)).initialize()
    definition = data_dir / "draft.yaml"
    write_draft(definition, model="jev-1.14.0")
    with definition.open("a") as output:
        output.write("question: Another question.\n")
    result = runner.invoke(app, ["graders", "new", str(definition)])
    assert result.exit_code == 1
    assert "Duplicate YAML key" in result.output
    assert Catalog(data_dir).list_graders() == []
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", "../escape"),
        ("version", 1),
        ("status", "active"),
        ("validation", None),
        ("question", ""),
        ("fields", ["unknown"]),
        ("yes_threshold", 0.2),
        ("no_threshold", True),
        ("origin", "other"),
        ("applies_to", ""),
        ("unknown", "value"),
    ],
)
def test_new_validates_entire_draft_before_resolving_model(environment, key, value):
    data_dir, _, factory = environment
    Ledger(Settings(data_dir)).initialize()
    definition = data_dir / "draft.yaml"
    write_draft(definition, **{key: value})
    result = runner.invoke(app, ["graders", "new", str(definition)])
    assert result.exit_code == 1, result.output
    assert Catalog(data_dir).list_graders() == []
    assert (data_dir / "ledger.jsonl").read_bytes() == b""
    factory.assert_not_called()


def test_jev_promotion_validation_and_retirement_use_catalog(environment):
    data_dir, _, factory = environment
    Ledger(Settings(data_dir)).initialize()
    definition = data_dir / "draft.yaml"
    write_draft(definition, model="jev-1.14.0")
    assert runner.invoke(app, ["graders", "new", str(definition)]).exit_code == 0
    rejected = runner.invoke(app, ["graders", "promote", "answer_complete", "--version", "1"])
    assert rejected.exit_code == 1 and "50 labels" in rejected.output
    catalog = Catalog(data_dir)
    [grader] = catalog.list_graders()
    catalog.save_validation(
        grader,
        Validation(
            labels=50,
            agreement=0.9,
            threshold_sweep=tuple(
                ThresholdValidation(no, yes, 0.9, None, None) for no, yes in threshold_pairs(grader)
            ),
            validated_at="2026-11-11T00:00:00+00:00",
            definition_hash=definition_fingerprint(grader),
            labels_hash="0" * 64,
        ),
    )
    promoted = runner.invoke(app, ["graders", "promote", "answer_complete", "--version", "1"])
    assert promoted.exit_code == 0, promoted.output
    assert json.loads(promoted.output)["status"] == "active"
    retired = runner.invoke(app, ["graders", "retire", "answer_complete", "--version", "1"])
    assert retired.exit_code == 0 and json.loads(retired.output)["status"] == "retired"
    repeated = runner.invoke(app, ["graders", "retire", "answer_complete", "--version", "1"])
    assert repeated.exit_code == 1
    factory.assert_not_called()


def test_builtin_code_promotion_executes_real_citation_tests(environment):
    data_dir, _, factory = environment
    Ledger(Settings(data_dir)).initialize()
    document = asdict(SEED_CODE)
    for key in ("version", "status", "validation"):
        document.pop(key)
    document.update(name="citations_valid", origin="human", fields=list(SEED_CODE.fields))
    Catalog(data_dir).create_draft(document)
    result = runner.invoke(app, ["graders", "promote", "citations_valid", "--version", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "active"
    factory.assert_not_called()


def test_external_code_promotion_requires_a_passing_test_target(environment, monkeypatch):
    data_dir, _, factory = environment
    Ledger(Settings(data_dir)).initialize()
    Catalog(data_dir).create_draft(
        {
            "name": "custom_check",
            "kind": "code",
            "applies_to": "rfi-agent",
            "fields": ["final_answer"],
            "function": "custom:check",
            "origin": "human",
        }
    )
    arguments = ["graders", "promote", "custom_check", "--version", "1"]
    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 1 and "--test-target" in missing.output
    target = data_dir / "test_custom.py"
    target.write_text("def test_custom():\n    assert False\n")
    (data_dir / "pytest.ini").write_text("[pytest]\naddopts=--collect-only\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    failed = runner.invoke(app, [*arguments, "--test-target", str(target)])
    assert failed.exit_code == 1 and "Code grader tests failed" in failed.output
    assert Catalog(data_dir).list_graders()[0].status == "draft"
    target.write_text("def test_custom():\n    assert True\n")
    passed = runner.invoke(app, [*arguments, "--test-target", str(target)])
    assert passed.exit_code == 0, passed.output
    assert Catalog(data_dir).list_graders()[0].status == "active"
    factory.assert_not_called()


def test_env_file_is_optional_and_environment_wins(tmp_path, monkeypatch):
    for name in ("HOPPER_DATA_DIR", "HOPPER_SPEND_CAP_USD", "JEV_USD_PER_MILLION_INPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    assert Settings.from_env(env_file).spend_cap_usd == Decimal("20")
    assert Settings.from_env(env_file).jev_usd_per_million_input_tokens == Decimal("0.042")
    env_file.write_text("HOPPER_SPEND_CAP_USD=3\nHOPPER_DATA_DIR=~/custom-hopper\n")
    assert Settings.from_env(env_file).spend_cap_usd == Decimal("3")
    assert Settings.from_env(env_file).data_dir == (tmp_path.home() / "custom-hopper")
    monkeypatch.setenv("HOPPER_SPEND_CAP_USD", "7")
    assert Settings.from_env(env_file).spend_cap_usd == Decimal("7")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "not-money"])
@pytest.mark.parametrize("name", ["HOPPER_SPEND_CAP_USD", "JEV_USD_PER_MILLION_INPUT_TOKENS"])
def test_invalid_money_is_rejected(value, name, tmp_path, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env(tmp_path / "missing.env")
