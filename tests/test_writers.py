"""Tests for session and extract writers."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from src.models import KnowledgeItem, KnowledgeType, Message, Role, Session, ToolUse
from src.normalizer import Normalizer
from src.writers.extract_writer import ExtractWriter
from src.writers.session_writer import SessionWriter
from tests.conftest import make_message, make_session


# ---------------------------------------------------------------------------
# SessionWriter
# ---------------------------------------------------------------------------

class TestSessionWriter:
    def test_creates_directory_structure(self, output_dir: Path) -> None:
        session = make_session(agent="claude-code")
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        assert path.parent == output_dir / "Sessions" / "claude-code"

    def test_file_contains_yaml_frontmatter(self, output_dir: Path) -> None:
        session = make_session()
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert content.startswith("---\n")
        assert "type: session-log" in content
        assert "agent: claude-code" in content
        assert "session_id: test-session-001" in content

    def test_frontmatter_contains_tags(self, output_dir: Path) -> None:
        session = make_session(project="my-proj")
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert "tags:" in content
        assert "  - claude-code" in content
        assert "  - session" in content

    def test_frontmatter_contains_date(self, output_dir: Path) -> None:
        session = make_session(
            timestamp=datetime(2025, 3, 15, tzinfo=timezone.utc),
        )
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert "date: 2025-03-15" in content

    def test_user_messages_section(self, output_dir: Path) -> None:
        session = make_session(
            messages=[
                make_message(Role.USER, "First question"),
                make_message(Role.ASSISTANT, "First answer"),
                make_message(Role.USER, "Second question"),
            ],
        )
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert "## User Messages" in content
        assert "First question" in content
        assert "Second question" in content

    def test_assistant_responses_section(self, output_dir: Path) -> None:
        session = make_session(
            messages=[
                make_message(Role.USER, "Hi"),
                make_message(Role.ASSISTANT, "Hello back"),
            ],
        )
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert "## Assistant Responses" in content
        assert "Hello back" in content

    def test_tool_usage_section(self, output_dir: Path) -> None:
        msg = make_message(
            Role.ASSISTANT,
            "Done",
            tool_uses=[ToolUse(name="Read", output="file contents")],
        )
        session = make_session(messages=[make_message(Role.USER, "go"), msg])
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert "## Tool Usage" in content
        assert "Read" in content

    def test_wikilinks_section(self, output_dir: Path) -> None:
        session = make_session(project="my-proj", agent="claude-code")
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert "## See Also" in content
        assert "[[projects/my-proj]]" in content
        assert "[[agents/claude-code]]" in content

    def test_append_only_raises_on_existing(self, output_dir: Path) -> None:
        session = make_session()
        writer = SessionWriter()
        writer.write_session(session, output_dir)
        with pytest.raises(FileExistsError):
            writer.write_session(session, output_dir)

    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        """Directories are created even if none exist yet."""
        session = make_session(agent="factory-droid")
        writer = SessionWriter()
        path = writer.write_session(session, tmp_path)
        assert path.exists()
        assert "factory-droid" in str(path)

    def test_write_extract_raises(self, output_dir: Path) -> None:
        session = make_session()
        writer = SessionWriter()
        with pytest.raises(NotImplementedError):
            writer.write_extract(session, [], output_dir)

    def test_total_messages_in_frontmatter(self, output_dir: Path) -> None:
        session = make_session(
            messages=[
                make_message(Role.USER, "a"),
                make_message(Role.ASSISTANT, "b"),
                make_message(Role.USER, "c"),
            ],
        )
        writer = SessionWriter()
        path = writer.write_session(session, output_dir)
        content = path.read_text()
        assert "total_messages: 3" in content


# ---------------------------------------------------------------------------
# ExtractWriter
# ---------------------------------------------------------------------------

class TestExtractWriter:
    def _make_items(self, session: Session) -> List[KnowledgeItem]:
        return Normalizer().normalize(session)

    def test_creates_directory_structure(self, output_dir: Path) -> None:
        session = make_session(agent="claude-code")
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        assert "agents/claude-code/extracts" in str(path)

    def test_filename_format(self, output_dir: Path) -> None:
        session = make_session(
            agent="codex",
            timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        assert path.name == "memory-extract-codex-2025-06-01.md"

    def test_yaml_frontmatter(self, output_dir: Path) -> None:
        session = make_session()
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        content = path.read_text()
        assert content.startswith("---\n")
        assert "type: memory-extract" in content
        assert "retention: inbox" in content
        assert "status: active" in content

    def test_resumo_section(self, output_dir: Path) -> None:
        session = make_session()
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        content = path.read_text()
        assert "## Resumo" in content

    def test_conteudo_migrado_section(self, output_dir: Path) -> None:
        session = make_session()
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        content = path.read_text()
        assert "## Conteudo migrado" in content

    def test_acompanhamento_section(self, output_dir: Path) -> None:
        session = make_session()
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        content = path.read_text()
        assert "## Acompanhamento" in content
        assert "- [ ] Review and categorize extract" in content

    def test_wikilinks_section(self, output_dir: Path) -> None:
        session = make_session(project="test-project", agent="claude-code")
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        content = path.read_text()
        assert "## Wikilinks relacionados" in content
        assert "[[agents/claude-code]]" in content

    def test_append_only_raises_on_existing(self, output_dir: Path) -> None:
        session = make_session()
        writer = ExtractWriter()
        items = self._make_items(session)
        writer.write_extract(session, items, output_dir)
        with pytest.raises(FileExistsError):
            writer.write_extract(session, items, output_dir)

    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        session = make_session(agent="factory-droid")
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, tmp_path)
        assert path.exists()

    def test_write_session_raises(self, output_dir: Path) -> None:
        session = make_session()
        writer = ExtractWriter()
        with pytest.raises(NotImplementedError):
            writer.write_session(session, output_dir)

    def test_session_id_in_frontmatter(self, output_dir: Path) -> None:
        session = make_session(id="my-sess-id")
        writer = ExtractWriter()
        items = self._make_items(session)
        path = writer.write_extract(session, items, output_dir)
        content = path.read_text()
        assert "session_id: my-sess-id" in content
