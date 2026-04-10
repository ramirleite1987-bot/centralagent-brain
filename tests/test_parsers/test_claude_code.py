"""Tests for Claude Code JSONL session parser."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings import AgentConfig, Settings
from src.models import Message, Role, Session, ToolUse
from src.parsers.claude_code import ClaudeCodeParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_with_path(source_path: Path) -> Settings:
    """Create Settings pointing claude-code at the given path."""
    s = Settings()
    s.agents["claude-code"] = AgentConfig(name="claude-code", source_path=source_path)
    return s


def _write_jsonl(path: Path, records: list) -> Path:
    """Write a list of dicts as JSONL to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


# ---------------------------------------------------------------------------
# discover_sessions
# ---------------------------------------------------------------------------

class TestDiscoverSessions:
    def test_discovers_jsonl_files(self, tmp_path: Path) -> None:
        proj = tmp_path / "projects" / "myproject"
        _write_jsonl(proj / "sess1.jsonl", [{"uuid": "1"}])
        _write_jsonl(proj / "sess2.jsonl", [{"uuid": "2"}])

        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "projects"))
        paths = parser.discover_sessions()
        assert len(paths) == 2
        assert all(p.suffix == ".jsonl" for p in paths)

    def test_empty_source_path(self, tmp_path: Path) -> None:
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "nonexistent"))
        assert parser.discover_sessions() == []


# ---------------------------------------------------------------------------
# parse_session – fixture file
# ---------------------------------------------------------------------------

class TestParseSessionFixture:
    def test_parses_fixture(self, claude_code_fixture: Path) -> None:
        settings = Settings()
        settings.agents["claude-code"] = AgentConfig(
            name="claude-code", source_path=claude_code_fixture.parent,
        )
        parser = ClaudeCodeParser(settings)
        session = parser.parse_session(claude_code_fixture)

        assert session.id == "abc-123"
        assert session.agent == "claude-code"
        assert len(session.messages) > 0

    def test_skips_meta_messages(self, claude_code_fixture: Path) -> None:
        settings = Settings()
        settings.agents["claude-code"] = AgentConfig(
            name="claude-code", source_path=claude_code_fixture.parent,
        )
        parser = ClaudeCodeParser(settings)
        session = parser.parse_session(claude_code_fixture)

        # The fixture has an isMeta message; it should not appear
        for msg in session.messages:
            assert "summary" not in msg.content.lower() or msg.role != Role.SYSTEM


# ---------------------------------------------------------------------------
# Content type handling
# ---------------------------------------------------------------------------

class TestContentTypes:
    def test_string_content(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": "Hello"}, "timestamp": "2025-01-15T10:00:00Z",
             "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "projects" / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "projects"))
        session = parser.parse_session(path)

        assert session.messages[0].content == "Hello"
        assert session.messages[0].role == Role.USER

    def test_array_content_with_text(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "assistant",
             "message": {"content": [{"type": "text", "text": "Hi there"}]},
             "timestamp": "2025-01-15T10:00:00Z", "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)

        assert session.messages[0].content == "Hi there"

    def test_array_content_with_tool_use(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "assistant",
             "message": {"content": [
                 {"type": "text", "text": "Creating file."},
                 {"type": "tool_use", "name": "Write", "input": {"path": "/tmp/a"}},
             ]},
             "timestamp": "2025-01-15T10:00:00Z", "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)

        msg = session.messages[0]
        assert msg.content == "Creating file."
        assert len(msg.tool_uses) == 1
        assert msg.tool_uses[0].name == "Write"

    def test_encrypted_content_skipped(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "assistant",
             "message": {"content": [
                 {"type": "text", "text": "Here you go"},
                 {"type": "encrypted_content", "data": "abc123"},
             ]},
             "timestamp": "2025-01-15T10:00:00Z", "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)

        assert session.messages[0].content == "Here you go"


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------

class TestMessageFiltering:
    def test_skips_unknown_roles(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "system",
             "message": {"content": "sys msg"}, "timestamp": "2025-01-15T10:00:00Z",
             "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_skips_empty_content(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": ""}, "timestamp": "2025-01-15T10:00:00Z",
             "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_skips_entries_without_uuid(self, tmp_path: Path) -> None:
        records = [
            {"role": "human", "message": {"content": "no uuid"},
             "timestamp": "2025-01-15T10:00:00Z", "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert len(session.messages) == 0


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_extracts_session_id(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": "hi"}, "timestamp": "2025-01-15T10:00:00Z",
             "sessionId": "my-session"},
        ]
        path = _write_jsonl(tmp_path / "projects" / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "projects"))
        session = parser.parse_session(path)
        assert session.id == "my-session"

    def test_extracts_project_name(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": "hi"}, "timestamp": "2025-01-15T10:00:00Z",
             "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "projects" / "my-project" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "projects"))
        session = parser.parse_session(path)
        assert session.project == "my-project"

    def test_extracts_timestamp(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": "hi"}, "timestamp": "2025-01-15T10:00:00Z",
             "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert session.timestamp == datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_malformed_jsonl_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "proj" / "s1.jsonl"
        path.parent.mkdir(parents=True)
        with open(path, "w") as f:
            f.write("NOT JSON\n")
            f.write(json.dumps({
                "uuid": "m1", "parentMessageId": "", "role": "human",
                "message": {"content": "valid"},
                "timestamp": "2025-01-15T10:00:00Z", "sessionId": "s1",
            }) + "\n")
            f.write("{broken json\n")

        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert len(session.messages) == 1
        assert session.messages[0].content == "valid"

    def test_invalid_timestamp_returns_none(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": "hi"}, "timestamp": "not-a-date",
             "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert session.messages[0].timestamp is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_session_file(self, tmp_path: Path) -> None:
        path = tmp_path / "proj" / "s1.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("")

        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert len(session.messages) == 0
        assert session.timestamp is None

    def test_parent_uuid_tree_ordering(self, tmp_path: Path) -> None:
        """Messages written out of order should be sorted by parent tree."""
        records = [
            {"uuid": "m2", "parentMessageId": "m1", "role": "assistant",
             "message": {"content": "response"}, "timestamp": "2025-01-15T10:00:05Z",
             "sessionId": "s1"},
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": "question"}, "timestamp": "2025-01-15T10:00:00Z",
             "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)

        assert session.messages[0].content == "question"
        assert session.messages[1].content == "response"

    def test_session_id_fallback_to_stem(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "human",
             "message": {"content": "hi"}, "timestamp": "2025-01-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "my-session.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert session.id == "my-session"

    def test_tool_result_content_in_output(self, tmp_path: Path) -> None:
        records = [
            {"uuid": "m1", "parentMessageId": "", "role": "assistant",
             "message": {"content": [
                 {"type": "tool_result", "tool_use_id": "t1", "content": "Success"},
             ]},
             "timestamp": "2025-01-15T10:00:00Z", "sessionId": "s1"},
        ]
        path = _write_jsonl(tmp_path / "proj" / "s1.jsonl", records)
        parser = ClaudeCodeParser(_settings_with_path(tmp_path / "proj"))
        session = parser.parse_session(path)
        assert len(session.messages) == 1
        assert "Success" in session.messages[0].content
