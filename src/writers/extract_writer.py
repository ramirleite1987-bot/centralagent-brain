"""Knowledge extract writer for the CentralAgent Brain ingestion pipeline.

Writes knowledge extract files to {vault}/agents/{agent}/extracts/memory-extract-{agent}-{date}.md
with YAML frontmatter containing retention metadata. Follows existing bilingual (Portuguese)
header convention. Respects append-only rule (never overwrites existing files).
"""

from pathlib import Path
from typing import List

from src.models import KnowledgeItem, KnowledgeType, Session
from src.normalizer import Normalizer
from src.writers.base import BaseWriter, WriterRegistry


@WriterRegistry.register
class ExtractWriter(BaseWriter):
    """Writes knowledge extracts to agents/{agent}/extracts/ in the Obsidian vault."""

    def __init__(self) -> None:
        self._normalizer = Normalizer()

    def write_session(self, session: Session, output_dir: Path) -> Path:
        """Not implemented for ExtractWriter. Use SessionWriter instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ExtractWriter does not write session logs. Use SessionWriter."
        )

    def write_extract(
        self,
        session: Session,
        items: List[KnowledgeItem],
        output_dir: Path,
    ) -> Path:
        """Write knowledge extracts from a session.

        Args:
            session: The source session.
            items: Knowledge items extracted from the session.
            output_dir: Root directory of the Obsidian vault.

        Returns:
            Path to the written file.

        Raises:
            FileExistsError: If the target file already exists (append-only).
        """
        date_str = self._normalizer._format_date(session.timestamp)
        agent = session.agent
        filename = f"memory-extract-{agent}-{date_str}.md"

        extracts_dir = output_dir / "agents" / agent / "extracts"
        extracts_dir.mkdir(parents=True, exist_ok=True)

        file_path = extracts_dir / filename

        # Avoid filename collisions by appending an incrementing suffix
        if file_path.exists():
            base = f"memory-extract-{agent}-{date_str}"
            counter = 2
            while True:
                file_path = extracts_dir / f"{base}-{counter}.md"
                if not file_path.exists():
                    break
                counter += 1

        content = self._render_extract(session, items, date_str)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def _render_extract(
        self,
        session: Session,
        items: List[KnowledgeItem],
        date_str: str,
    ) -> str:
        """Render knowledge extract as Markdown with YAML frontmatter."""
        agent = session.agent
        lines: List[str] = []

        # YAML frontmatter
        lines.append("---")
        lines.append(f'title: "memory-extract-{agent}-{date_str}"')
        lines.append("type: memory-extract")
        lines.append(f"agent: {agent}")
        lines.append("source: session-log")
        lines.append("category: technical")
        lines.append("status: active")
        lines.append("retention: inbox")
        lines.append("tags:")
        lines.append("  - extract")
        lines.append(f"  - {agent}")
        if session.project:
            lines.append(f"  - {session.project}")
        lines.append(f"createdAt: {date_str}")
        lines.append(f"date: {date_str}")
        lines.append(f"session_id: {session.id}")
        lines.append("---")
        lines.append("")

        # Resumo (Summary) - bilingual Portuguese convention
        lines.append("## Resumo")
        summaries = [i for i in items if i.type == KnowledgeType.SUMMARY]
        if summaries:
            for item in summaries:
                lines.append(item.content)
        else:
            lines.append(f"Knowledge extract from {agent} session on {date_str}.")
        lines.append("")

        # Conteudo migrado (Migrated Content)
        lines.append("## Conteudo migrado")
        non_summary = [i for i in items if i.type != KnowledgeType.SUMMARY]
        if non_summary:
            for item in non_summary:
                lines.append(f"### {item.title}")
                lines.append(item.content)
                lines.append("")
        else:
            lines.append("No additional knowledge items extracted.")
            lines.append("")

        # Acompanhamento (Follow-up)
        lines.append("## Acompanhamento")
        lines.append("- [ ] Review and categorize extract")
        lines.append("- [ ] Archive after retention period (30 days inbox)")
        lines.append("")

        # Wikilinks relacionados
        wikilinks = self._normalizer.session_wikilinks(session)
        lines.append("## Wikilinks relacionados")
        if wikilinks:
            for link in wikilinks:
                lines.append(f"- {link}")
        else:
            lines.append(f"- [[agents/{agent}]]")
        lines.append("")

        return "\n".join(lines)
