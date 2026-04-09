"""Claude Code JSONL session parser.

Parses session files from ~/.claude/projects/{name}/{uuid}.jsonl.
Handles dual content types (string vs array of text objects),
skips isMeta messages and encrypted_content blocks, and resolves
parentUuid tree into chronological order.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from config.settings import AgentConfig, Settings
from src.models import Message, Role, Session, ToolUse
from src.parsers.base import BaseParser, ParserRegistry


@ParserRegistry.register
class ClaudeCodeParser(BaseParser):
    """Parser for Claude Code JSONL session files."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        agent_cfg: AgentConfig = self._settings.agents["claude-code"]
        self._source_path: Path = agent_cfg.source_path or Path("~/.claude/projects").expanduser()

    @property
    def agent_name(self) -> str:
        return "claude-code"

    def discover_sessions(self) -> List[Path]:
        """Discover all .jsonl session files under the source path."""
        if not self._source_path.exists():
            return []
        sessions = sorted(self._source_path.glob("**/*.jsonl"))
        return sessions

    def parse_session(self, path: Path) -> Session:
        """Parse a single JSONL session file into a Session object."""
        messages_by_uuid: Dict[str, Dict[str, Any]] = {}
        session_id: Optional[str] = None

        for raw_message in self._stream_jsonl(path):
            # Skip isMeta messages (summaries, etc.)
            if raw_message.get("isMeta", False):
                continue

            uuid = raw_message.get("uuid")
            if not uuid:
                continue

            if session_id is None:
                session_id = raw_message.get("sessionId", path.stem)

            messages_by_uuid[uuid] = raw_message

        # Resolve parent tree into chronological order
        ordered = self._resolve_message_order(messages_by_uuid)

        # Parse each message
        messages: List[Message] = []
        for raw in ordered:
            parsed = self._parse_message(raw)
            if parsed is not None:
                messages.append(parsed)

        # Extract project name from path: ~/.claude/projects/{name}/{uuid}.jsonl
        project = self._extract_project_name(path)

        # Determine session timestamp from first message
        timestamp = messages[0].timestamp if messages else None

        return Session(
            id=session_id or path.stem,
            agent="claude-code",
            timestamp=timestamp,
            project=project,
            messages=messages,
            source_path=path,
        )

    def _stream_jsonl(self, path: Path) -> Generator[Dict[str, Any], None, None]:
        """Stream JSONL file line-by-line, skipping malformed lines."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _resolve_message_order(
        self, messages: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Resolve parentUuid/parentMessageId tree into chronological order.

        Builds a tree from parent references, then walks it depth-first
        to produce an ordered list. Falls back to timestamp sorting if
        tree resolution fails.
        """
        if not messages:
            return []

        # Build children map
        children: Dict[Optional[str], List[str]] = {}
        for uuid, msg in messages.items():
            parent = msg.get("parentMessageId") or msg.get("parentUuid") or ""
            children.setdefault(parent, []).append(uuid)

        # Sort children by timestamp at each level
        def sort_key(uuid: str) -> str:
            return messages[uuid].get("timestamp", "")

        for parent in children:
            children[parent].sort(key=sort_key)

        # Walk tree from roots
        ordered: List[Dict[str, Any]] = []
        roots = children.get("", [])
        if not roots:
            # Fallback: sort all by timestamp
            return sorted(messages.values(), key=lambda m: m.get("timestamp", ""))

        stack = list(reversed(roots))
        while stack:
            uuid = stack.pop()
            ordered.append(messages[uuid])
            child_uuids = children.get(uuid, [])
            stack.extend(reversed(child_uuids))

        return ordered

    def _parse_message(self, raw: Dict[str, Any]) -> Optional[Message]:
        """Parse a raw JSONL message dict into a Message object."""
        role_str = raw.get("role", "")
        if role_str == "human":
            role = Role.USER
        elif role_str == "assistant":
            role = Role.ASSISTANT
        else:
            return None

        message_data = raw.get("message", {})
        content_raw = message_data.get("content", "")

        content, tool_uses = self._extract_content(content_raw)

        # Skip messages with no usable content
        if not content and not tool_uses:
            return None

        timestamp = self._parse_timestamp(raw.get("timestamp"))

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
        - Array: list of content blocks (text, tool_use, tool_result, encrypted_content)
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
            elif block_type == "tool_result":
                result_content = block.get("content", "")
                tool_id = block.get("tool_use_id", "")
                # Attach result to matching tool_use if possible
                if isinstance(result_content, str) and result_content:
                    text_parts.append(f"[Tool Result: {result_content}]")
            elif block_type == "encrypted_content":
                # Skip encrypted thinking blocks
                continue

        return "\n".join(text_parts), tool_uses

    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse an ISO 8601 timestamp string."""
        if not ts:
            return None
        try:
            # Handle Z suffix
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    def _extract_project_name(self, path: Path) -> Optional[str]:
        """Extract project name from session file path.

        Expected path: ~/.claude/projects/{project_name}/{uuid}.jsonl
        """
        try:
            parts = path.parts
            # Find 'projects' in path and take the next part
            for i, part in enumerate(parts):
                if part == "projects" and i + 1 < len(parts) - 1:
                    return parts[i + 1]
        except (IndexError, ValueError):
            pass
        return None
