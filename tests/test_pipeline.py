"""Tests for the pipeline orchestrator, including integration tests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from src.dedup import DedupTracker
from src.models import Message, Role, Session
from src.pipeline import AgentResult, Pipeline, PipelineResult
from tests.conftest import make_message, make_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path: Path, vault_path: Path = None) -> Pipeline:
    """Create a Pipeline with temp state and vault dirs."""
    if vault_path is None:
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        (vault_path / "Sessions").mkdir()
        (vault_path / "agents").mkdir()
    settings = Settings()
    settings.state_dir = tmp_path / "state"
    settings.vault_path = vault_path
    return Pipeline(settings=settings)


def _make_claude_session_file(sessions_dir: Path, session_id: str = "abc123") -> Path:
    """Write a minimal Claude Code JSONL session file."""
    session_file = sessions_dir / f"{session_id}.jsonl"
    lines = [
        json.dumps({
            "parentUuid": None,
            "uuid": "msg-1",
            "type": "human",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2025-01-15T10:00:00Z",
        }),
        json.dumps({
            "parentUuid": "msg-1",
            "uuid": "msg-2",
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
            "timestamp": "2025-01-15T10:00:01Z",
        }),
    ]
    session_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session_file


# ---------------------------------------------------------------------------
# AgentResult / PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_total_discovered(self) -> None:
        result = PipelineResult()
        result.agents["a"] = AgentResult(agent="a", discovered=3)
        result.agents["b"] = AgentResult(agent="b", discovered=5)
        assert result.total_discovered == 8

    def test_total_written(self) -> None:
        result = PipelineResult()
        result.agents["a"] = AgentResult(agent="a", written=2)
        assert result.total_written == 2

    def test_total_errors(self) -> None:
        result = PipelineResult()
        result.agents["a"] = AgentResult(agent="a", errors=["e1", "e2"])
        assert result.total_errors == 2

    def test_empty_result(self) -> None:
        result = PipelineResult()
        assert result.total_discovered == 0
        assert result.total_written == 0
        assert result.total_errors == 0


# ---------------------------------------------------------------------------
# Pipeline._resolve_agents
# ---------------------------------------------------------------------------

class TestResolveAgents:
    def test_unknown_agent_raises(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        with pytest.raises(ValueError, match="Unknown agent"):
            pipeline._resolve_agents(["nonexistent-agent"])

    def test_valid_agent_returned(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        result = pipeline._resolve_agents(["claude-code"])
        assert result == ["claude-code"]

    def test_none_returns_enabled(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        result = pipeline._resolve_agents(None)
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Dedup integration with pipeline
# ---------------------------------------------------------------------------

class TestPipelineDedup:
    def test_dedup_skips_already_exported(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        # Pre-mark a session as exported
        pipeline._dedup.mark_exported("claude-code", "already-done")
        assert pipeline._dedup.is_exported("claude-code", "already-done")

    def test_force_resets_dedup(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        pipeline._dedup.mark_exported("claude-code", "s1")
        # Simulate force behavior
        pipeline._dedup.reset("claude-code")
        assert not pipeline._dedup.is_exported("claude-code", "s1")


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_does_not_write_files(self, tmp_path: Path) -> None:
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        (vault_path / "Sessions").mkdir()
        (vault_path / "agents").mkdir()
        pipeline = _make_pipeline(tmp_path, vault_path=vault_path)

        # Mock parser to return a session
        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = [Path("/fake/session.jsonl")]
        session = make_session(id="dry-run-test", agent="claude-code")
        mock_parser.parse_session.return_value = session

        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run(agents=["claude-code"], dry_run=True)

        agent_result = result.agents["claude-code"]
        assert agent_result.discovered == 1
        assert agent_result.written == 1  # counted as "would write"
        assert result.dry_run is True

        # Verify no dedup state saved (dry-run shouldn't mark exported)
        assert not pipeline._dedup.is_exported("claude-code", "dry-run-test")

    def test_dry_run_flag_in_result(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = []
        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run(agents=["claude-code"], dry_run=True)
        assert result.dry_run is True


# ---------------------------------------------------------------------------
# Full pipeline flow (parse -> normalize -> write) with mocks
# ---------------------------------------------------------------------------

class TestFullPipelineFlow:
    def test_parse_normalize_write(self, tmp_path: Path) -> None:
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        (vault_path / "Sessions").mkdir()
        (vault_path / "agents").mkdir()
        pipeline = _make_pipeline(tmp_path, vault_path=vault_path)

        session = make_session(
            id="full-flow-test",
            agent="claude-code",
            messages=[
                make_message(Role.USER, "Implement feature X"),
                make_message(Role.ASSISTANT, "I'll implement feature X by modifying module Y. This is a key decision."),
            ],
        )

        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = [Path("/fake/session.jsonl")]
        mock_parser.parse_session.return_value = session

        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run(agents=["claude-code"])

        agent_result = result.agents["claude-code"]
        assert agent_result.discovered == 1
        assert agent_result.parsed == 1
        assert agent_result.written == 1
        assert agent_result.skipped_error == 0

        # Verify dedup marks session
        assert pipeline._dedup.is_exported("claude-code", "full-flow-test")

    def test_empty_session_skipped(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        session = make_session(id="empty-test", agent="claude-code", messages=[])

        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = [Path("/fake/session.jsonl")]
        mock_parser.parse_session.return_value = session

        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run(agents=["claude-code"])

        assert result.agents["claude-code"].skipped_empty == 1
        assert result.agents["claude-code"].written == 0


# ---------------------------------------------------------------------------
# Multi-agent batch
# ---------------------------------------------------------------------------

class TestMultiAgentBatch:
    def test_multiple_agents(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)

        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = []

        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run(agents=["claude-code", "codex"])

        assert "claude-code" in result.agents
        assert "codex" in result.agents

    def test_all_enabled_agents(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)

        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = []

        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run()

        assert len(result.agents) > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestPipelineErrorHandling:
    def test_parse_error_counted(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)

        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = [Path("/fake/session.jsonl")]
        mock_parser.parse_session.side_effect = RuntimeError("Parse failed")

        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run(agents=["claude-code"])

        agent_result = result.agents["claude-code"]
        assert agent_result.skipped_error == 1
        assert len(agent_result.errors) == 1
        assert "Parse failed" in agent_result.errors[0]

    def test_discovery_error_handled(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)

        mock_parser = MagicMock()
        mock_parser.discover_sessions.side_effect = OSError("No such dir")

        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            result = pipeline.run(agents=["claude-code"])

        agent_result = result.agents["claude-code"]
        assert len(agent_result.errors) == 1
        assert "Discovery failed" in agent_result.errors[0]

    def test_write_error_handled(self, tmp_path: Path) -> None:
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        (vault_path / "Sessions").mkdir()
        (vault_path / "agents").mkdir()
        pipeline = _make_pipeline(tmp_path, vault_path=vault_path)

        session = make_session(id="write-fail", agent="claude-code")
        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = [Path("/fake/s.jsonl")]
        mock_parser.parse_session.return_value = session

        with patch.object(pipeline, "_get_parser", return_value=mock_parser), \
             patch.object(pipeline._session_writer, "write_session", side_effect=OSError("Disk full")):
            result = pipeline.run(agents=["claude-code"])

        agent_result = result.agents["claude-code"]
        assert agent_result.skipped_error == 1
        assert "Disk full" in agent_result.errors[0]

    def test_no_parser_handled(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)

        with patch.object(pipeline, "_get_parser", side_effect=KeyError("no parser")):
            result = pipeline.run(agents=["claude-code"])

        agent_result = result.agents["claude-code"]
        assert len(agent_result.errors) == 1


# ---------------------------------------------------------------------------
# list_sessions and status
# ---------------------------------------------------------------------------

class TestListAndStatus:
    def test_list_sessions(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = [
            Path("/fake/session-a.jsonl"),
            Path("/fake/session-b.jsonl"),
        ]
        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            sessions = pipeline.list_sessions("claude-code")
        assert len(sessions) == 2
        assert sessions[0].session_id == "session-a"

    def test_status_returns_all_agents(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        mock_parser = MagicMock()
        mock_parser.discover_sessions.return_value = []
        with patch.object(pipeline, "_get_parser", return_value=mock_parser):
            status = pipeline.status()
        assert len(status) > 0
        for name, info in status.items():
            assert info.agent == name
