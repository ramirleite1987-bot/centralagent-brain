"""Tests for Cursor SQLite session parser."""

import json
import sqlite3
from pathlib import Path

import pytest

from config.settings import AgentConfig, Settings
from src.models import Message, Role
from src.parsers.cursor import CursorParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_with_path(source_path: Path) -> Settings:
    s = Settings()
    s.agents["cursor"] = AgentConfig(name="cursor", source_path=source_path)
    return s


def _create_store_db(
    db_path: Path,
    composer_data: dict | None = None,
    create_table: bool = True,
) -> Path:
    """Create a Cursor store.db with optional composer data."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    if create_table:
        cursor.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
        if composer_data is not None:
            cursor.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                ("composerData", json.dumps(composer_data)),
            )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# discover_sessions
# ---------------------------------------------------------------------------

class TestDiscoverSessions:
    def test_discovers_store_db(self, tmp_path: Path) -> None:
        _create_store_db(tmp_path / "chats" / "ws1" / "store.db")
        _create_store_db(tmp_path / "chats" / "ws2" / "store.db")

        parser = CursorParser(_settings_with_path(tmp_path / "chats"))
        paths = parser.discover_sessions()
        assert len(paths) == 2

    def test_empty_source_path(self, tmp_path: Path) -> None:
        parser = CursorParser(_settings_with_path(tmp_path / "nonexistent"))
        assert parser.discover_sessions() == []


# ---------------------------------------------------------------------------
# parse_session – fixture
# ---------------------------------------------------------------------------

class TestParseSessionFixture:
    def test_parses_cursor_store_db(self, cursor_store_db: Path) -> None:
        settings = _settings_with_path(cursor_store_db.parent)
        parser = CursorParser(settings)
        session = parser.parse_session(cursor_store_db)

        assert session.agent == "cursor"
        assert session.id == "test-composer-1"
        assert len(session.messages) == 4

    def test_message_roles(self, cursor_store_db: Path) -> None:
        settings = _settings_with_path(cursor_store_db.parent)
        parser = CursorParser(settings)
        session = parser.parse_session(cursor_store_db)

        roles = [m.role for m in session.messages]
        assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


# ---------------------------------------------------------------------------
# Content type handling
# ---------------------------------------------------------------------------

class TestContentTypes:
    def test_bubble_type_1_is_user(self, tmp_path: Path) -> None:
        data = {"c1": {"conversation": [{"type": 1, "text": "user msg"}]}}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert session.messages[0].role == Role.USER

    def test_bubble_type_2_is_assistant(self, tmp_path: Path) -> None:
        data = {"c1": {"conversation": [{"type": 2, "text": "assistant msg"}]}}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert session.messages[0].role == Role.ASSISTANT


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------

class TestMessageFiltering:
    def test_unknown_bubble_type_skipped(self, tmp_path: Path) -> None:
        data = {"c1": {"conversation": [
            {"type": 1, "text": "user"},
            {"type": 99, "text": "unknown"},
            {"type": 2, "text": "assistant"},
        ]}}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert len(session.messages) == 2

    def test_empty_text_skipped(self, tmp_path: Path) -> None:
        data = {"c1": {"conversation": [
            {"type": 1, "text": ""},
            {"type": 2, "text": "response"},
        ]}}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert len(session.messages) == 1


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_extracts_composer_name_as_project(self, tmp_path: Path) -> None:
        data = {"c1": {"name": "My Project", "conversation": [{"type": 1, "text": "hi"}]}}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert session.project == "My Project"

    def test_extracts_created_at(self, tmp_path: Path) -> None:
        data = {"c1": {
            "createdAt": "2025-01-15T10:00:00Z",
            "conversation": [{"type": 1, "text": "hi"}],
        }}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert session.timestamp is not None


# ---------------------------------------------------------------------------
# Error handling – locked/corrupt databases
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_table_returns_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "store.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other (x TEXT)")
        conn.close()

        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db_path)
        assert len(session.messages) == 0

    def test_invalid_json_in_db_returns_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "store.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO cursorDiskKV VALUES (?, ?)", ("composerData", "NOT JSON"))
        conn.commit()
        conn.close()

        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db_path)
        assert len(session.messages) == 0

    def test_corrupt_db_file_returns_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "store.db"
        db_path.write_text("this is not a sqlite database")

        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db_path)
        assert len(session.messages) == 0

    def test_no_composer_data_key(self, tmp_path: Path) -> None:
        db = _create_store_db(tmp_path / "store.db", composer_data=None)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert len(session.messages) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_conversation_list(self, tmp_path: Path) -> None:
        data = {"c1": {"conversation": []}}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert len(session.messages) == 0

    def test_non_dict_bubble_skipped(self, tmp_path: Path) -> None:
        data = {"c1": {"conversation": ["not a dict", {"type": 1, "text": "ok"}]}}
        db = _create_store_db(tmp_path / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert len(session.messages) == 1

    def test_no_composers_in_data(self, tmp_path: Path) -> None:
        db = _create_store_db(tmp_path / "store.db", composer_data={})
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        assert len(session.messages) == 0

    def test_project_fallback_to_parent_name(self, tmp_path: Path) -> None:
        data = {"c1": {"conversation": [{"type": 1, "text": "hi"}]}}
        db = _create_store_db(tmp_path / "myworkspace" / "store.db", data)
        parser = CursorParser(_settings_with_path(tmp_path))
        session = parser.parse_session(db)
        # name not set, falls back to parent dir name
        assert session.project == "myworkspace"
