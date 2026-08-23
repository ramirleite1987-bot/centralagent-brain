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
        self._source_path: Path = (
            agent_cfg.source_path or Path("~/.codex/sessions").expanduser()
        )

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

            has_payload = "payload" in entry and isinstance(entry["payload"], dict)
            payload = entry.get("payload", {}) if has_payload else entry

            if entry_type == "session_meta":
                session_id = payload.get("id") or payload.get("session_id", path.stem)
                metadata["model"] = payload.get("model") or payload.get(
                    "model_provider"
                )
                metadata["version"] = payload.get("version") or payload.get(
                    "cli_version"
                )
                cwd = payload.get("cwd")
                if cwd:
                    project = Path(cwd).name
                continue

            if entry_type in ("response_item", "event_msg", "user_message"):
                parsed = self._parse_entry(entry_type, payload)
                if parsed is not None:
                    messages.append(parsed)

        timestamp = messages[0].timestamp if messages else None

        return Session(
            id=session_id or path.stem,
            agent="codex",
            timestamp=timestamp,
            project=project,
            messages=messages,
            source_path=path,
            metadata=metadata,
        )

    def _parse_entry(
        self, entry_type: str, payload: Dict[str, Any]
    ) -> Optional[Message]:
        """Parse a JSONL entry payload into a Message object.

        Filters out XML-wrapped system messages (content starting with <).
        """
        if entry_type == "event_msg":
            msg_type = payload.get("type", "")
            if msg_type == "user_message":
                content = payload.get("message", "")
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content if c)
                if not content or self._is_xml_system_message(content):
                    return None
                timestamp = self._parse_timestamp(payload.get("timestamp"))
                return Message(role=Role.USER, content=content, timestamp=timestamp)
            return None

        if entry_type == "user_message":
            content = payload.get("content", "")
            if not content or self._is_xml_system_message(content):
                return None
            timestamp = self._parse_timestamp(payload.get("timestamp"))
            return Message(role=Role.USER, content=content, timestamp=timestamp)

        if entry_type == "response_item":
            role_str = payload.get("role", "")
            if role_str == "user":
                role = Role.USER
            elif role_str in ("assistant", "developer"):
                role = Role.ASSISTANT
            else:
                return None

            content_raw = payload.get("content", "")
            content, tool_uses = self._extract_content(content_raw)

            if self._is_xml_system_message(content):
                return None

            if not content and not tool_uses:
                return None

            timestamp = self._parse_timestamp(payload.get("timestamp"))

            return Message(
                role=role,
                content=content,
                timestamp=timestamp,
                tool_uses=tool_uses,
            )

        return None

    def _is_xml_system_message(self, content: str) -> bool:
        """Check if content is an XML-wrapped system message."""
        return content.strip().startswith("<")

    def _extract_content(self, content_raw: Any) -> Tuple[str, List[ToolUse]]:
        """Extract text content and tool uses from message content.

        Handles multiple formats:
        - String: plain text content
        - Array: list of content blocks (text, tool_use, output_text, input_text)
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

            if block_type in ("text", "output_text", "input_text"):
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
