"""Session log writer for the CentralAgent Brain ingestion pipeline.

Writes session log files to {vault}/Sessions/{agent-type}/{date}-{project}-{topic}.md
with YAML frontmatter containing all required fields. Creates directories as needed.
Respects append-only rule (never overwrites existing files).
"""

from pathlib import Path
from typing import List

from src.models import KnowledgeItem, Message, Role, Session
from src.normalizer import Normalizer
from src.writers.base import BaseWriter, WriterRegistry


@WriterRegistry.register
class SessionWriter(BaseWriter):
    """Writes session logs to Sessions/{agent-type}/ in the Obsidian vault."""

    def __init__(self) -> None:
        self._normalizer = Normalizer()

    def write_session(self, session: Session, output_dir: Path) -> Path:
        """Write a session log to the output directory.

        Args:
            session: The normalized session to write.
            output_dir: Root directory of the Obsidian vault.

        Returns:
            Path to the written file.

        Raises:
            FileExistsError: If the target file already exists (append-only).
        """
        slug = self._normalizer.session_slug(session)
        sessions_dir = output_dir / "Sessions" / session.agent
        sessions_dir.mkdir(parents=True, exist_ok=True)

        file_path = sessions_dir / f"{slug}.md"

        # Append-only: never overwrite existing files
        if file_path.exists():
            raise FileExistsError(
                f"Session file already exists (append-only): {file_path}"
            )

        content = self._render_session(session)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def write_extract(
        self,
        session: Session,
        items: List[KnowledgeItem],
        output_dir: Path,
    ) -> Path:
        """Not implemented for SessionWriter. Use ExtractWriter instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SessionWriter does not write extracts. Use ExtractWriter."
        )

    def _render_session(self, session: Session) -> str:
        """Render a session as Markdown with YAML frontmatter."""
        lines: List[str] = []

        # YAML frontmatter
        lines.append("---")
        lines.append("type: session-log")
        lines.append(f"date: {self._normalizer._format_date(session.timestamp)}")
        project = session.project or "unknown"
        lines.append(f"project: {project}")

        # Tags
        tags = self._normalizer.session_tags(session)
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {tag}")

        lines.append(f"session_id: {session.id}")
        lines.append(f"agent: {session.agent}")

        # Optional fields
        if session.source_path:
            lines.append(f"source_file: {session.source_path}")
        if session.cwd:
            lines.append(f"cwd: {session.cwd}")
        if session.model:
            lines.append(f"model: {session.model}")

        total_messages = len(session.messages)
        lines.append(f"total_messages: {total_messages}")
        lines.append("---")
        lines.append("")

        # Title
        date_str = self._normalizer._format_date(session.timestamp)
        lines.append(f"# Session: {project} - {date_str}")
        lines.append("")

        # User Messages
        user_msgs = [m for m in session.messages if m.role == Role.USER]
        lines.append("## User Messages")
        lines.append("")
        if user_msgs:
            for i, msg in enumerate(user_msgs, 1):
                lines.append(f"### Message {i}")
                lines.append(msg.content.strip())
                lines.append("")
        else:
            lines.append("No user messages recorded.")
            lines.append("")

        # Assistant Responses
        assistant_msgs = [m for m in session.messages if m.role == Role.ASSISTANT]
        lines.append("## Assistant Responses")
        lines.append("")
        if assistant_msgs:
            for i, msg in enumerate(assistant_msgs, 1):
                lines.append(f"### Response {i}")
                lines.append(msg.content.strip())
                lines.append("")
        else:
            lines.append("No assistant responses recorded.")
            lines.append("")

        # Tool Usage
        tool_uses = []
        for msg in session.messages:
            for tool in msg.tool_uses:
                tool_uses.append(tool)

        lines.append("## Tool Usage")
        if tool_uses:
            for tool in tool_uses:
                desc = tool.output[:80] if tool.output else "invoked"
                lines.append(f"- {tool.name}: {desc}")
        else:
            lines.append("No tool usage recorded.")
        lines.append("")

        # Source
        lines.append("## Source")
        if session.source_path:
            lines.append(f"Session file: `{session.source_path}`")
        else:
            lines.append(f"Session ID: {session.id}")
        lines.append("")

        # See Also (wikilinks)
        wikilinks = self._normalizer.session_wikilinks(session)
        if wikilinks:
            lines.append("## See Also")
            for link in wikilinks:
                lines.append(f"- {link}")
            lines.append("")

        return "\n".join(lines)
