"""Factory Droid JSONL session parser.

Parses session files from ~/.factory/sessions/{name}/{uuid}.jsonl.
Reads companion {uuid}.settings.json for model and project metadata.
Extracts messages from type:'message' entries, metadata from session_start
and settings file. Extracts todo_state entries as task progress artifacts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import AgentConfig, Settings
from src.models import Message, Role, Session, ToolUse
from src.parsers.base import BaseParser, ParserRegistry


@ParserRegistry.register
class FactoryDroidParser(BaseParser):
    """Parser for Factory Droid JSONL session files."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        agent_cfg: AgentConfig = self._settings.agents["factory-droid"]
        self._source_path: Path = agent_cfg.source_path or Path("~/.factory/sessions").expanduser()

    @property
    def agent_name(self) -> str:
        return "factory-droid"

    def discover_sessions(self) -> List[Path]:
        """Discover all .jsonl session files under the source path."""
        if not self._source_path.exists():
            return []
        sessions = sorted(self._source_path.glob("**/*.jsonl"))
        return sessions

    def parse_session(self, path: Path) -> Session:
        """Parse a single JSONL session file into a Session object."""
        messages: List[Message] = []
        session_id: Optional[str] = None
        project: Optional[str] = None
        metadata: Dict[str, Any] = {}
        todo_items: List[Dict[str, str]] = []

        # Read companion settings file
        settings_path = path.with_suffix(".settings.json")
        settings_data = self._read_settings(settings_path)
        if settings_data:
            metadata["model"] = settings_data.get("model")
            metadata["version"] = settings_data.get("version")
            project = settings_data.get("project_name")

        for entry in self._stream_jsonl(path):
            entry_type = entry.get("type", "")

            if entry_type == "session_start":
                session_id = entry.get("session_id", path.stem)
                cwd = entry.get("cwd")
                if cwd and not project:
                    project = Path(cwd).name
                continue

            if entry_type == "message":
                parsed = self._parse_message(entry)
                if parsed is not None:
                    messages.append(parsed)
                continue

            if entry_type == "todo_state":
                items = entry.get("items", [])
                todo_items = [
                    {"title": item.get("title", ""), "status": item.get("status", "")}
                    for item in items
                    if isinstance(item, dict)
                ]

        if todo_items:
            metadata["todo_state"] = todo_items

        timestamp = messages[0].timestamp if messages else None

        return Session(
            id=session_id or path.stem,
            agent="factory-droid",
            timestamp=timestamp,
            project=project,
            messages=messages,
            metadata=metadata,
            source_path=path,
        )

    def _read_settings(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read companion settings JSON file if it exists."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _parse_message(self, entry: Dict[str, Any]) -> Optional[Message]:
        """Parse a message entry into a Message object."""
        role_str = entry.get("role", "")
        if role_str == "user":
            role = Role.USER
        elif role_str == "assistant":
            role = Role.ASSISTANT
        else:
            return None

        content_raw = entry.get("content", "")
        content, tool_uses = self._extract_content(content_raw)

        if not content and not tool_uses:
            return None

        timestamp = self._parse_timestamp(entry.get("timestamp"))

        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            tool_uses=tool_uses,
        )

    def _extract_content(self, content_raw: Any) -> Tuple[str, List[ToolUse]]:
        """Extract text content and tool uses from message content.

        Handles two formats:
        - String: plain text content
        - Array: list of content blocks (text, tool_use)
        """
        if isinstance(content_raw, str):
            return content_raw, []

        if not isinstance(content_raw, list):
            return "", []

        text_parts: List[str] = []
        tool_uses: List[ToolUse] = []

        for block in content_raw:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_uses.append(
                    ToolUse(
                        name=block.get("name", "unknown"),
                        input=block.get("input"),
                    )
                )

        return "\n\n".join(text_parts), tool_uses

