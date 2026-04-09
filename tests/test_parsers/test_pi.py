"""Tests for Pi stub adapter parser."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings import AgentConfig, Settings
from src.models import Message, Role
from src.parsers.pi import PiParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_with_path(source_path: Path) -> Settings:
    s = Settings()
    s.agents["pi"] = AgentConfig(name="pi", source_path=source_path)
    return s


# ---------------------------------------------------------------------------
# discover_sessions
# ---------------------------------------------------------------------------

class TestDiscoverSessions:
    def test_discovers_json_and_txt(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.txt").write_text("hello")
        (tmp_path / "c.py").write_text("# not this")

        parser = PiParser(_settings_with_path(tmp_path))
        paths = parser.discover_sessions()
        assert len(paths) == 2
        names = {p.name for p in paths}
        assert names == {"a.json", "b.txt"}

    def test_empty_source_path(self, tmp_path: Path) -> None:
        parser = PiParser(_settings_with_path(tmp_path / "nonexistent"))
        assert parser.discover_sessions() == []


# ---------------------------------------------------------------------------
# parse_session – fixture
# ---------------------------------------------------------------------------

class TestParseSessionFixture:
    def test_parses_json_fixture(self, pi_fixture: Path) -> None:
        settings = _settings_with_path(pi_fixture.parent)
        parser = PiParser(settings)
        session = parser.parse_session(pi_fixture)

        assert session.id == "pi-session-001"
        assert session.agent == "pi"
        assert session.project == "test-project"
        assert len(session.messages) == 4

    def test_message_roles(self, pi_fixture: Path) -> None:
        settings = _settings_with_path(pi_fixture.parent)
        parser = PiParser(settings)
        session = parser.parse_session(pi_fixture)

        roles = [m.role for m in session.messages]
        assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


# ---------------------------------------------------------------------------
# Content type handling – JSON vs text
# ---------------------------------------------------------------------------

class TestContentTypes:
    def test_json_file_parsed(self, tmp_path: Path) -> None:
        data = {
            "session_id": "s1",
            "messages": [
                {"role": "user", "content": "hi", "timestamp": "2025-06-01T10:00:00Z"},
            ],
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.messages[0].content == "hi"
        assert session.messages[0].role == Role.USER

    def test_text_file_becomes_single_user_message(self, tmp_path: Path) -> None:
        path = tmp_path / "note.txt"
        path.write_text("My conversation notes here")

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 1
        assert session.messages[0].role == Role.USER
        assert session.messages[0].content == "My conversation notes here"


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------

class TestMessageFiltering:
    def test_unknown_role_skipped(self, tmp_path: Path) -> None:
        data = {"messages": [{"role": "system", "content": "sys"}]}
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_empty_content_skipped(self, tmp_path: Path) -> None:
        data = {"messages": [{"role": "user", "content": ""}]}
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_extracts_session_id_and_project(self, tmp_path: Path) -> None:
        data = {
            "session_id": "my-session",
            "project": "my-project",
            "messages": [{"role": "user", "content": "hi"}],
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.id == "my-session"
        assert session.project == "my-project"

    def test_timestamp_from_first_message(self, tmp_path: Path) -> None:
        data = {
            "messages": [
                {"role": "user", "content": "hi", "timestamp": "2025-06-01T10:00:00Z"},
            ],
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.timestamp == datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_text_file_session_id_is_stem(self, tmp_path: Path) -> None:
        path = tmp_path / "my-note.txt"
        path.write_text("content")

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.id == "my-note"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_malformed_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("NOT JSON")

        parser = PiParser(_settings_with_path(tmp_path))
        with pytest.raises(ValueError, match="Failed to parse Pi JSON"):
            parser.parse_session(path)

    def test_invalid_timestamp_returns_none(self, tmp_path: Path) -> None:
        data = {
            "messages": [
                {"role": "user", "content": "hi", "timestamp": "not-a-date"},
            ],
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.messages[0].timestamp is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("")

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_json_with_no_messages_key(self, tmp_path: Path) -> None:
        path = tmp_path / "s1.json"
        path.write_text(json.dumps({"session_id": "s1"}))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert len(session.messages) == 0

    def test_session_id_fallback_to_stem(self, tmp_path: Path) -> None:
        data = {"messages": [{"role": "user", "content": "hi"}]}
        path = tmp_path / "fallback-id.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.id == "fallback-id"

    def test_naive_timestamp_gets_utc(self, tmp_path: Path) -> None:
        data = {
            "messages": [
                {"role": "user", "content": "hi", "timestamp": "2025-06-01T10:00:00"},
            ],
        }
        path = tmp_path / "s1.json"
        path.write_text(json.dumps(data))

        parser = PiParser(_settings_with_path(tmp_path))
        session = parser.parse_session(path)
        assert session.messages[0].timestamp.tzinfo is not None
