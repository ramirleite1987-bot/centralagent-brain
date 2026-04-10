"""Pi stub adapter for manually-dropped import files.

Reads JSON or plain text files from a configurable directory
(default: ~/.centralagent/pi-import/). Placeholder until Pi
provides an export API.

Expected JSON format:
{
  "session_id": "...",
  "project": "...",
  "messages": [
    {"role": "user"|"assistant", "content": "...", "timestamp": "ISO8601"}
  ]
}

Plain text files are treated as a single user message.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import AgentConfig, Settings
from src.models import Message, Role, Session
from src.parsers.base import BaseParser, ParserRegistry


@ParserRegistry.register
class PiParser(BaseParser):
    """Parser for manually-imported Pi conversation files."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        agent_cfg: AgentConfig = self._settings.agents["pi"]
        self._source_path: Path = (
            agent_cfg.source_path
            or Path("~/.centralagent/pi-import").expanduser()
        )

    @property
    def agent_name(self) -> str:
        return "pi"

    def discover_sessions(self) -> List[Path]:
        """Discover all .json and .txt files under the import directory."""
        if not self._source_path.exists():
            return []
        json_files = list(self._source_path.glob("**/*.json"))
        txt_files = list(self._source_path.glob("**/*.txt"))
        return sorted(json_files + txt_files)

    def parse_session(self, path: Path) -> Session:
        """Parse a single import file into a Session object.

        JSON files are parsed for structured message data.
        Plain text files are wrapped as a single user message.

        Args:
            path: Path to the import file.

        Returns:
            A normalized Session object.

        Raises:
            ValueError: If the file cannot be parsed.
        """
        if path.suffix == ".json":
            return self._parse_json(path)
        return self._parse_text(path)

    def _parse_json(self, path: Path) -> Session:
        """Parse a JSON import file into a Session."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Failed to parse Pi JSON file {path}: {exc}") from exc

        session_id = data.get("session_id", path.stem)
        project = data.get("project")
        raw_messages = data.get("messages", [])

        messages: List[Message] = []
        for raw in raw_messages:
            parsed = self._parse_message(raw)
            if parsed is not None:
                messages.append(parsed)

        timestamp = messages[0].timestamp if messages else None

        return Session(
            id=session_id,
            agent="pi",
            timestamp=timestamp,
            project=project,
            messages=messages,
            source_path=path,
        )

    def _parse_text(self, path: Path) -> Session:
        """Parse a plain text file as a single user message."""
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"Failed to read Pi text file {path}: {exc}") from exc

        if not content:
            return Session(
                id=path.stem,
                agent="pi",
                messages=[],
                source_path=path,
            )

        message = Message(role=Role.USER, content=content)

        return Session(
            id=path.stem,
            agent="pi",
            messages=[message],
            source_path=path,
        )

    def _parse_message(self, raw: Dict[str, Any]) -> Optional[Message]:
        """Parse a single message dict from the JSON import."""
        role_str = raw.get("role", "")
        if role_str == "user":
            role = Role.USER
        elif role_str == "assistant":
            role = Role.ASSISTANT
        else:
            return None

        content = raw.get("content", "")
        if not content:
            return None

        timestamp = self._parse_timestamp(raw.get("timestamp"))

        return Message(role=role, content=content, timestamp=timestamp)

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        """Parse an ISO 8601 timestamp string."""
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
