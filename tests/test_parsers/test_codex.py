"""Tests for OpenAI Codex JSONL session parser."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings import AgentConfig, Settings
from src.models import Message, Role, ToolUse
from src.parsers.codex import CodexParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_with_path(source_path: Path) -> Settings:
    s = Settings()
    s.agents["codex"] = AgentConfig(name="codex", source_path=source_path)
    return s


def _write_jsonl(path: Path, records: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


# ---------------------------------------------------------------------------
# discover_sessions
# ---------------------------------------------------------------------------

class TestDiscoverSessions:
    def test_discovers_rollout_files(self, tmp_path: Path) -> None:
        d = tmp_path / "2025" / "02" / "10"
        _write_jsonl(d / "rollout-001.jsonl", [{"type": "session_meta"}])
        _write_jsonl(d / "rollout-002.jsonl", [{"type": "session_meta"}])
        # Non-matching file should be ignored
        _write_jsonl(d / "other.jsonl", [{}])

        parser = CodexParser(_settings_with_path(tmp_path))
        paths = parser.discover_sessions()
        assert len(paths) == 2
        assert all("rollout-" in p.name for p in paths)

    def test_empty_source_path(self, tmp_path: Path) -> None:
        parser = CodexParser(_settings_with_path(tmp_path / "nonexistent"))
        assert parser.discover_sessions() == []


# ---------------------------------------------------------------------------
# parse_session – fixture file
# ---------------------------------------------------------------------------

class TestParseSessionFixture:
    def test_parses_fixture(self, codex_fixture: Path) -> None:
        settings = _settings_with_path(codex_fixture.parent)
        parser = CodexParser(settings)
        session = parser.parse_session(codex_fixture)

        assert session.id == "codex-session-001"
        assert session.agent == "codex"
        assert session.project == "project"
        assert len(session.messages) > 0

    def test_filters_xml_system_messages(self, codex_fixture: Path) -> None:
        settings = _settings_with_path(codex_fixture.parent)
        parser = CodexParser(settings)
        session = parser.parse_session(codex_fixture)

        for msg in session.messages:
            assert not msg.content.strip().startswith("<")


# ---------------------------------------------------------------------------
# Content type handling
# ---------------------------------------------------------------------------

class TestContentTypes:
    def test_string_content(self, tmp_path: Path) -> None:
        records = [
            {"type": "user_message", "content": "Hello", "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)

        assert session.messages[0].content == "Hello"
        assert session.messages[0].role == Role.USER

    def test_array_content_with_tool_use(self, tmp_path: Path) -> None:
        records = [
            {"type": "response_item", "role": "assistant",
             "content": [
                 {"type": "text", "text": "Patching..."},
                 {"type": "tool_use", "name": "apply_patch", "input": {"patch": "diff"}},
             ],
             "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)

        msg = session.messages[0]
        assert msg.content == "Patching..."
        assert len(msg.tool_uses) == 1
        assert msg.tool_uses[0].name == "apply_patch"


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------

class TestMessageFiltering:
    def test_xml_system_messages_filtered(self, tmp_path: Path) -> None:
        records = [
            {"type": "response_item", "role": "assistant",
             "content": "<system>\nYou are helpful.\n</system>",
             "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_xml_user_messages_filtered(self, tmp_path: Path) -> None:
        records = [
            {"type": "user_message", "content": "<context>data</context>",
             "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_agent_reasoning_and_turn_context_skipped(self, tmp_path: Path) -> None:
        records = [
            {"type": "agent_reasoning", "content": "thinking...", "timestamp": "2025-02-10T14:00:00Z"},
            {"type": "turn_context", "context": {"files_read": ["a.py"]}, "timestamp": "2025-02-10T14:00:01Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_empty_content_skipped(self, tmp_path: Path) -> None:
        records = [
            {"type": "response_item", "role": "assistant", "content": "",
             "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_session_meta_extracts_id_and_project(self, tmp_path: Path) -> None:
        records = [
            {"type": "session_meta", "session_id": "sess-42", "cwd": "/home/user/myproj",
             "model": "o4-mini", "version": "0.1.2"},
            {"type": "user_message", "content": "hi", "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)

        assert session.id == "sess-42"
        assert session.project == "myproj"

    def test_timestamp_from_first_message(self, tmp_path: Path) -> None:
        records = [
            {"type": "user_message", "content": "hi", "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.timestamp == datetime(2025, 2, 10, 14, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_malformed_jsonl_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "rollout-1.jsonl"
        with open(path, "w") as f:
            f.write("BROKEN\n")
            f.write(json.dumps({"type": "user_message", "content": "ok",
                                "timestamp": "2025-02-10T14:00:00Z"}) + "\n")
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 1

    def test_invalid_timestamp(self, tmp_path: Path) -> None:
        records = [
            {"type": "user_message", "content": "hi", "timestamp": "nope"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.messages[0].timestamp is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_session_file(self, tmp_path: Path) -> None:
        path = tmp_path / "rollout-1.jsonl"
        path.write_text("")
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0
        assert session.timestamp is None

    def test_session_id_fallback_to_stem(self, tmp_path: Path) -> None:
        records = [
            {"type": "user_message", "content": "hi", "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-abc.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.id == "rollout-abc"

    def test_unknown_role_skipped(self, tmp_path: Path) -> None:
        records = [
            {"type": "response_item", "role": "system", "content": "sys",
             "timestamp": "2025-02-10T14:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "rollout-1.jsonl", records)
        parser = CodexParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0
