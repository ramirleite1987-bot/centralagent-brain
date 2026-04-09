"""Cursor SQLite session parser.

Queries ~/.cursor/chats/*/store.db and legacy
~/.config/Cursor/User/workspaceStorage/*/state.vscdb.
Reads cursorDiskKV table, parses composerData and bubbleId keys.
Opens databases read-only to avoid WAL conflicts. Handles locked
databases gracefully.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import AgentConfig, Settings
from src.models import Message, Role, Session, ToolUse
from src.parsers.base import BaseParser, ParserRegistry


@ParserRegistry.register
class CursorParser(BaseParser):
    """Parser for Cursor SQLite session databases."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        agent_cfg: AgentConfig = self._settings.agents["cursor"]
        self._source_path: Path = agent_cfg.source_path or Path("~/.cursor/chats").expanduser()
        self._legacy_path: Path = Path("~/.config/Cursor/User/workspaceStorage").expanduser()

    @property
    def agent_name(self) -> str:
        return "cursor"

    def discover_sessions(self) -> List[Path]:
        """Discover all store.db and state.vscdb files under source paths."""
        dbs: List[Path] = []

        for base, pattern in [
            (self._source_path, "**/store.db"),
            (self._legacy_path, "**/state.vscdb"),
        ]:
            if base.exists():
                dbs.extend(sorted(base.glob(pattern)))

        return dbs

    def parse_session(self, path: Path) -> Session:
        """Parse a single Cursor SQLite database into Session objects.

        Each database may contain multiple composer conversations stored
        under the composerData key in the cursorDiskKV table. This method
        extracts the first (or only) composer conversation found.
        """
        composers = self._read_composer_data(path)

        if not composers:
            return Session(
                id=path.parent.name,
                agent="cursor",
                messages=[],
                source_path=path,
            )

        # Use the first composer conversation
        composer_id, composer = next(iter(composers.items()))
        messages = self._parse_conversation(composer)

        project = composer.get("name") or path.parent.name
        timestamp = messages[0].timestamp if messages else None
        created_at = self._parse_timestamp(composer.get("createdAt"))

        return Session(
            id=composer_id,
            agent="cursor",
            timestamp=created_at or timestamp,
            project=project,
            messages=messages,
            source_path=path,
        )

    def _read_composer_data(self, path: Path) -> Dict[str, Any]:
        """Read composerData from cursorDiskKV table.

        Opens database in read-only mode to avoid WAL conflicts.
        Returns empty dict if database is locked or table doesn't exist.
        """
        uri = f"file:{path}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5)
        except sqlite3.OperationalError:
            return {}

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM cursorDiskKV WHERE key = ?",
                ("composerData",),
            )
            row = cursor.fetchone()
            if not row:
                return {}

            data = json.loads(row[0])
            if isinstance(data, dict):
                return data
            return {}
        except (sqlite3.OperationalError, sqlite3.DatabaseError, json.JSONDecodeError):
            return {}
        finally:
            conn.close()

    def _parse_conversation(self, composer: Dict[str, Any]) -> List[Message]:
        """Parse a composer conversation into a list of Messages.

        Cursor stores conversation as a list of bubbles with type:
        - 1 = user message
        - 2 = assistant message
        """
        conversation = composer.get("conversation", [])
        messages: List[Message] = []

        for bubble in conversation:
            if not isinstance(bubble, dict):
                continue

            bubble_type = bubble.get("type")
            text = bubble.get("text", "")

            if not text:
                continue

            if bubble_type == 1:
                role = Role.USER
            elif bubble_type == 2:
                role = Role.ASSISTANT
            else:
                continue

            messages.append(
                Message(
                    role=role,
                    content=text,
                    timestamp=None,
                )
            )

        return messages

    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse an ISO-8601 timestamp string."""
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
