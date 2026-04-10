#!/usr/bin/env python3
"""Generate the cursor_store.db test fixture.

Run: python3 tests/fixtures/create_cursor_fixture.py
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "cursor_store.db"


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)"
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
        "INSERT OR REPLACE INTO cursorDiskKV VALUES (?, ?)",
        ("composerData", json.dumps(composer_data)),
    )
    conn.commit()
    conn.close()
    print(f"Created {DB_PATH}")


if __name__ == "__main__":
    main()
