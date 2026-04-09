"""Tests for Factory Droid JSONL session parser."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings import AgentConfig, Settings
from src.models import Message, Role, ToolUse
from src.parsers.factory_droid import FactoryDroidParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_with_path(source_path: Path) -> Settings:
    s = Settings()
    s.agents["factory-droid"] = AgentConfig(name="factory-droid", source_path=source_path)
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
    def test_discovers_jsonl_files(self, tmp_path: Path) -> None:
        _write_jsonl(tmp_path / "sessions" / "proj" / "s1.jsonl", [{}])
        _write_jsonl(tmp_path / "sessions" / "proj" / "s2.jsonl", [{}])

        parser = FactoryDroidParser(_settings_with_path(tmp_path / "sessions"))
        assert len(parser.discover_sessions()) == 2

    def test_empty_source_path(self, tmp_path: Path) -> None:
        parser = FactoryDroidParser(_settings_with_path(tmp_path / "nonexistent"))
        assert parser.discover_sessions() == []


# ---------------------------------------------------------------------------
# parse_session – fixture
# ---------------------------------------------------------------------------

class TestParseSessionFixture:
    def test_parses_fixture(self, factory_droid_fixture: Path) -> None:
        settings = _settings_with_path(factory_droid_fixture.parent)
        parser = FactoryDroidParser(settings)
        session = parser.parse_session(factory_droid_fixture)

        assert session.id == "abc-123"
        assert session.agent == "factory-droid"
        assert len(session.messages) > 0

    def test_reads_settings_file(self, factory_droid_fixture: Path) -> None:
        settings = _settings_with_path(factory_droid_fixture.parent)
        parser = FactoryDroidParser(settings)
        session = parser.parse_session(factory_droid_fixture)
        # Settings file has project_name
        assert session.project == "centralagent-brain"

    def test_extracts_todo_state(self, factory_droid_fixture: Path) -> None:
        settings = _settings_with_path(factory_droid_fixture.parent)
        parser = FactoryDroidParser(settings)
        session = parser.parse_session(factory_droid_fixture)
        assert "todo_state" in session.metadata
        assert len(session.metadata["todo_state"]) == 2


# ---------------------------------------------------------------------------
# Content type handling
# ---------------------------------------------------------------------------

class TestContentTypes:
    def test_string_content(self, tmp_path: Path) -> None:
        records = [
            {"type": "message", "role": "user", "content": "Hello",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.messages[0].content == "Hello"

    def test_array_content_with_tool_use(self, tmp_path: Path) -> None:
        records = [
            {"type": "message", "role": "assistant",
             "content": [
                 {"type": "text", "text": "Writing file."},
                 {"type": "tool_use", "name": "write_file", "input": {"path": "a.py"}},
             ],
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)

        msg = session.messages[0]
        assert msg.content == "Writing file."
        assert len(msg.tool_uses) == 1
        assert msg.tool_uses[0].name == "write_file"


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------

class TestMessageFiltering:
    def test_session_start_not_a_message(self, tmp_path: Path) -> None:
        records = [
            {"type": "session_start", "session_id": "s1", "cwd": "/tmp"},
            {"type": "message", "role": "user", "content": "hi",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 1

    def test_unknown_role_skipped(self, tmp_path: Path) -> None:
        records = [
            {"type": "message", "role": "system", "content": "sys",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_empty_content_skipped(self, tmp_path: Path) -> None:
        records = [
            {"type": "message", "role": "user", "content": "",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_session_start_provides_id_and_project(self, tmp_path: Path) -> None:
        records = [
            {"type": "session_start", "session_id": "sess-42",
             "cwd": "/home/user/myproj"},
            {"type": "message", "role": "user", "content": "hi",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.id == "sess-42"
        assert session.project == "myproj"

    def test_settings_file_overrides_cwd_project(self, tmp_path: Path) -> None:
        records = [
            {"type": "session_start", "session_id": "s1", "cwd": "/tmp/other"},
            {"type": "message", "role": "user", "content": "hi",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        # Write companion settings
        settings_path = path.with_suffix(".settings.json")
        settings_path.write_text(json.dumps({"project_name": "from-settings"}))

        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.project == "from-settings"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_malformed_jsonl_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "s1.jsonl"
        with open(path, "w") as f:
            f.write("NOT JSON\n")
            f.write(json.dumps({"type": "message", "role": "user",
                                "content": "ok", "timestamp": "2025-06-15T10:00:00Z"}) + "\n")
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 1

    def test_invalid_settings_json_ignored(self, tmp_path: Path) -> None:
        records = [
            {"type": "message", "role": "user", "content": "hi",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        path.with_suffix(".settings.json").write_text("NOT JSON")

        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 1

    def test_missing_settings_file_ok(self, tmp_path: Path) -> None:
        records = [
            {"type": "message", "role": "user", "content": "hi",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_session_file(self, tmp_path: Path) -> None:
        path = tmp_path / "s1.jsonl"
        path.write_text("")
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0
        assert session.timestamp is None

    def test_session_id_fallback_to_stem(self, tmp_path: Path) -> None:
        records = [
            {"type": "message", "role": "user", "content": "hi",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "my-session.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.id == "my-session"

    def test_todo_state_with_non_dict_items(self, tmp_path: Path) -> None:
        records = [
            {"type": "todo_state", "items": ["not a dict", {"title": "ok", "status": "done"}]},
            {"type": "message", "role": "user", "content": "hi",
             "timestamp": "2025-06-15T10:00:00Z"},
        ]
        path = _write_jsonl(tmp_path / "s1.jsonl", records)
        parser = FactoryDroidParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.metadata.get("todo_state", [])) == 1
