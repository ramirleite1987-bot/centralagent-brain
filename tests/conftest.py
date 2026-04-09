"""Shared test fixtures for centralagent-brain tests."""

import json
import sqlite3
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
