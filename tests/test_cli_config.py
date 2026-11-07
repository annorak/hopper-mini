from decimal import Decimal

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from hopper.cli import app
from hopper.config import DATA_DIRECTORIES, Settings

runner = CliRunner()


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
    stub = runner.invoke(app, ["graders"])
    assert stub.exit_code == 2
    assert "not implemented" in stub.output


def test_init_preserves_existing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    assert runner.invoke(app, ["init"]).exit_code == 0
    assert all((tmp_path / directory).is_dir() for directory in DATA_DIRECTORIES)
    ledger = tmp_path / "ledger.jsonl"
    assert ledger.read_bytes() == b""
    ledger.write_text("existing ledger\n")
    span = tmp_path / "spans" / "existing.parquet"
    span.write_bytes(b"existing data")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "10 data directories ready" in result.output
    assert ledger.read_text() == "existing ledger\n"
    assert span.read_bytes() == b"existing data"


def test_env_file_is_optional_and_environment_wins(tmp_path, monkeypatch):
    for name in ("HOPPER_DATA_DIR", "HOPPER_SPEND_CAP_USD", "JEV_USD_PER_1K_QUESTIONS"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    assert Settings.from_env(env_file).spend_cap_usd == Decimal("20")
    env_file.write_text("HOPPER_SPEND_CAP_USD=3\nHOPPER_DATA_DIR=~/custom-hopper\n")
    assert Settings.from_env(env_file).spend_cap_usd == Decimal("3")
    assert Settings.from_env(env_file).data_dir == (tmp_path.home() / "custom-hopper")
    monkeypatch.setenv("HOPPER_SPEND_CAP_USD", "7")
    assert Settings.from_env(env_file).spend_cap_usd == Decimal("7")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "not-money"])
@pytest.mark.parametrize("name", ["HOPPER_SPEND_CAP_USD", "JEV_USD_PER_1K_QUESTIONS"])
def test_invalid_money_is_rejected(value, name, tmp_path, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env(tmp_path / "missing.env")
