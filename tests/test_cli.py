"""Tests for CLI entrypoint and Typer commands."""

from __future__ import annotations

from typer.testing import CliRunner

from episodeforge.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "EpisodeForge: Enterprise Robotics Data Pipeline" in result.output
    assert "generate" in result.output
    assert "spec" in result.output
    assert "deliver" in result.output


def test_cli_spec_lint_valid():
    result = runner.invoke(app, ["spec", "lint", "configs/clients/client_a_lerobot.v1.yaml"])
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_cli_spec_diff():
    result = runner.invoke(
        app,
        [
            "spec",
            "diff",
            "configs/clients/client_a_lerobot.v1.yaml",
            "configs/clients/client_a_lerobot.v2.yaml",
        ],
    )
    assert result.exit_code == 0
    assert "Spec Diff" in result.output
    assert "target_fps" in result.output
    assert "rot6d" in result.output
