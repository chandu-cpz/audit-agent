from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from audit_agent import cli


def test_demo_runs_offline_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "FIXTURES_DIR", tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["demo", "--n", "3", "--seed", "42"])

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "sources").exists()
    assert (tmp_path / "primary_outputs").exists()
    assert (tmp_path / "audit_reports.json").exists()
    assert len(list((tmp_path / "sources").glob("*.md"))) == 3
    assert len(list((tmp_path / "primary_outputs").glob("*.json"))) == 6
    assert "Business Impact Catch Rate" in result.stdout


def test_demo_replaces_stale_fixtures_between_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "FIXTURES_DIR", tmp_path)

    runner = CliRunner()
    first = runner.invoke(cli.app, ["demo", "--n", "3", "--seed", "42"])
    second = runner.invoke(cli.app, ["demo", "--n", "1", "--seed", "42"])

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    assert len(list((tmp_path / "sources").glob("*.md"))) == 1
    assert len(list((tmp_path / "primary_outputs").glob("*.json"))) == 2