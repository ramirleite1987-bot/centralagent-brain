"""Shared test fixtures for centralagent-brain tests."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from src.models import Message, Role, Session, ToolUse

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def make_message(
    role: Role = Role.USER,
    content: str = "test message",
    timestamp: Optional[datetime] = None,
    tool_uses: Optional[List[ToolUse]] = None,
) -> Message:
    """Create a Message with sensible defaults."""
    return Message(
        role=role,
        content=content,
        timestamp=timestamp or datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        tool_uses=tool_uses or [],
    )


def make_session(
    id: str = "test-session-001",
    agent: str = "claude-code",
    messages: Optional[List[Message]] = None,
    timestamp: Optional[datetime] = None,
    project: Optional[str] = "test-project",
    cwd: Optional[str] = "/home/user/project",
    model: Optional[str] = None,
) -> Session:
    """Create a Session with sensible defaults."""
    if messages is None:
        messages = [
            make_message(Role.USER, "Hello"),
            make_message(Role.ASSISTANT, "Hi there!"),
        ]
    return Session(
        id=id,
        agent=agent,
        timestamp=timestamp or datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        project=project,
        cwd=cwd,
        model=model,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Fixture file paths
# ---------------------------------------------------------------------------

@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def claude_code_fixture() -> Path:
    """Path to the sample Claude Code JSONL session."""
    return FIXTURES_DIR / "claude_code_session.jsonl"


@pytest.fixture
def codex_fixture() -> Path:
    """Path to the sample Codex JSONL session."""
    return FIXTURES_DIR / "codex_session.jsonl"


@pytest.fixture
def factory_droid_fixture() -> Path:
    """Path to the sample Factory Droid JSONL session."""
    return FIXTURES_DIR / "factory_droid_session.jsonl"


@pytest.fixture
def pi_fixture() -> Path:
    """Path to the sample Pi manual import JSON."""
    return FIXTURES_DIR / "pi_manual_import.json"


# ---------------------------------------------------------------------------
# Cursor SQLite fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cursor_store_db(tmp_path: Path) -> Path:
    """Create a temporary Cursor store.db with sample composer data."""
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)"
    )

    composer_data = {
        "test-composer-1": {
            "composerId": "test-composer-1",
            "name": "Test Session",
            "createdAt": "2025-01-15T10:00:00Z",
            "conversation": [
                {"type": 1, "text": "Hello, help me with Python", "bubbleId": "b1"},
                {"type": 2, "text": "Sure, I can help with Python!", "bubbleId": "b2"},
                {"type": 1, "text": "Write a function", "bubbleId": "b3"},
                {"type": 2, "text": "Here is a function:", "bubbleId": "b4"},
            ],
        }
    }

    cursor.execute(
        "INSERT INTO cursorDiskKV VALUES (?, ?)",
        ("composerData", json.dumps(composer_data)),
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory mimicking an Obsidian vault."""
    sessions_dir = tmp_path / "Sessions"
    sessions_dir.mkdir()
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_session() -> Session:
    """A ready-made session for tests that don't need to customize it."""
    return make_session()
