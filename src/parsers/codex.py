"""OpenAI Codex JSONL session parser.

Parses session files from ~/.codex/sessions/{Y}/{M}/{D}/rollout-*.jsonl.
Extracts messages from response_item and user_message entries, reads
session_meta for metadata, filters XML-wrapped system messages (content
starting with <). Extracts agent_reasoning and turn_context for enrichment.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import AgentConfig, Settings
from src.models import Message, Role, Session, ToolUse
from src.parsers.base import BaseParser, ParserRegistry


@ParserRegistry.register
class CodexParser(BaseParser):
    """Parser for OpenAI Codex JSONL session files."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        agent_cfg: AgentConfig = self._settings.agents["codex"]
        self._source_path: Path = agent_cfg.source_path or Path("~/.codex/sessions").expanduser()

    @property
    def agent_name(self) -> str:
        return "codex"

    def discover_sessions(self) -> List[Path]:
        """Discover all rollout-*.jsonl session files under the source path."""
        if not self._source_path.exists():
            return []
        sessions = sorted(self._source_path.glob("**/rollout-*.jsonl"))
        return sessions

    def parse_session(self, path: Path) -> Session:
        """Parse a single Codex JSONL session file into a Session object."""
        messages: List[Message] = []
        session_id: Optional[str] = None
        project: Optional[str] = None
        metadata: Dict[str, Any] = {}

        for entry in self._stream_jsonl(path):
            entry_type = entry.get("type", "")

            if entry_type == "session_meta":
                session_id = entry.get("session_id", path.stem)
                metadata["model"] = entry.get("model")
                metadata["version"] = entry.get("version")
                cwd = entry.get("cwd")
                if cwd:
                    project = Path(cwd).name
                continue

            if entry_type in ("response_item", "user_message"):
                parsed = self._parse_entry(entry)
                if parsed is not None:
                    messages.append(parsed)

            # agent_reasoning and turn_context are skipped as messages
            # but could be used for enrichment in the future

        timestamp = messages[0].timestamp if messages else None

        return Session(
            id=session_id or path.stem,
            agent="codex",
            timestamp=timestamp,
            project=project,
            messages=messages,
            source_path=path,
        )

    def _parse_entry(self, entry: Dict[str, Any]) -> Optional[Message]:
        """Parse a JSONL entry into a Message object.

        Filters out XML-wrapped system messages (content starting with <).
        """
        entry_type = entry.get("type", "")

        if entry_type == "user_message":
            content = entry.get("content", "")
            if not content or self._is_xml_system_message(content):
                return None
            timestamp = self._parse_timestamp(entry.get("timestamp"))
            return Message(role=Role.USER, content=content, timestamp=timestamp)

        # response_item
        role_str = entry.get("role", "")
        if role_str == "user":
            role = Role.USER
        elif role_str == "assistant":
            role = Role.ASSISTANT
        else:
            return None

        content_raw = entry.get("content", "")
        content, tool_uses = self._extract_content(content_raw)

        # Filter XML-wrapped system messages
        if self._is_xml_system_message(content):
            return None

        if not content and not tool_uses:
            return None

        timestamp = self._parse_timestamp(entry.get("timestamp"))

        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            tool_uses=tool_uses,
        )

    def _is_xml_system_message(self, content: str) -> bool:
        """Check if content is an XML-wrapped system message."""
        return content.strip().startswith("<")

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

