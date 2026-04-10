"""Tests for the normalization engine (slugify, wikilinks, Normalizer)."""

from datetime import datetime, timezone
from typing import List

import pytest

from src.models import KnowledgeType, Message, Role, Session, ToolUse
from src.normalizer import Normalizer, make_wikilink, slugify
from tests.conftest import make_message, make_session


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic_lowercasing(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_accents_stripped(self) -> None:
        assert slugify("café résumé") == "cafe-resume"

    def test_special_chars_removed(self) -> None:
        assert slugify("hello! @world# $test%") == "hello-world-test"

    def test_multiple_spaces_collapsed(self) -> None:
        assert slugify("hello   world") == "hello-world"

    def test_multiple_hyphens_collapsed(self) -> None:
        assert slugify("hello---world") == "hello-world"

    def test_leading_trailing_hyphens_stripped(self) -> None:
        assert slugify("--hello--") == "hello"

    def test_unicode_normalization(self) -> None:
        # ñ, ü, ö should be stripped of diacritics
        assert slugify("niño über König") == "nino-uber-konig"

    def test_empty_string(self) -> None:
        assert slugify("") == ""

    def test_only_special_chars(self) -> None:
        assert slugify("!!!@@@###") == ""

    def test_numbers_preserved(self) -> None:
        assert slugify("version 2.0 release") == "version-20-release"

    def test_mixed_accents_and_special(self) -> None:
        assert slugify("Ça fait très bien!") == "ca-fait-tres-bien"


# ---------------------------------------------------------------------------
# make_wikilink
# ---------------------------------------------------------------------------

class TestMakeWikilink:
    def test_basic_wikilink(self) -> None:
        assert make_wikilink("projects/my-proj") == "[[projects/my-proj]]"

    def test_wikilink_with_spaces(self) -> None:
        assert make_wikilink("some note") == "[[some note]]"

    def test_empty_string(self) -> None:
        assert make_wikilink("") == "[[]]"


# ---------------------------------------------------------------------------
# Normalizer.session_slug
# ---------------------------------------------------------------------------

class TestSessionSlug:
    def test_format(self) -> None:
        session = make_session(
            project="My Project",
            timestamp=datetime(2025, 3, 15, tzinfo=timezone.utc),
        )
        normalizer = Normalizer()
        slug = normalizer.session_slug(session)
        assert slug.startswith("2025-03-15-my-project-")

    def test_no_project(self) -> None:
        session = make_session(project=None)
        normalizer = Normalizer()
        slug = normalizer.session_slug(session)
        assert "-unknown-" in slug

    def test_topic_from_first_user_message(self) -> None:
        session = make_session(
            messages=[make_message(Role.USER, "Fix the login bug")],
        )
        normalizer = Normalizer()
        slug = normalizer.session_slug(session)
        assert "fix-the-login-bug" in slug


# ---------------------------------------------------------------------------
# Normalizer.session_wikilinks
# ---------------------------------------------------------------------------

class TestSessionWikilinks:
    def test_with_project(self) -> None:
        session = make_session(project="My Project", agent="claude-code")
        normalizer = Normalizer()
        links = normalizer.session_wikilinks(session)
        assert "[[projects/my-project]]" in links
        assert "[[agents/claude-code]]" in links

    def test_without_project(self) -> None:
        session = make_session(project=None, agent="codex")
        normalizer = Normalizer()
        links = normalizer.session_wikilinks(session)
        assert len(links) == 1
        assert "[[agents/codex]]" in links


# ---------------------------------------------------------------------------
# Normalizer.session_tags
# ---------------------------------------------------------------------------

class TestSessionTags:
    def test_tags_include_agent_and_session(self) -> None:
        session = make_session(agent="claude-code", project=None)
        normalizer = Normalizer()
        tags = normalizer.session_tags(session)
        assert "claude-code" in tags
        assert "session" in tags

    def test_tags_include_project_slug(self) -> None:
        session = make_session(project="My Project")
        normalizer = Normalizer()
        tags = normalizer.session_tags(session)
        assert "my-project" in tags


# ---------------------------------------------------------------------------
# Normalizer.normalize – knowledge extraction
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_produces_summary(self) -> None:
        session = make_session()
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        summaries = [i for i in items if i.type == KnowledgeType.SUMMARY]
        assert len(summaries) == 1

    def test_no_summary_without_user_messages(self) -> None:
        session = make_session(
            messages=[make_message(Role.ASSISTANT, "Hello")]
        )
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        summaries = [i for i in items if i.type == KnowledgeType.SUMMARY]
        assert len(summaries) == 0

    def test_extracts_tool_patterns(self) -> None:
        msg = make_message(
            Role.ASSISTANT,
            "Done",
            tool_uses=[ToolUse(name="Read"), ToolUse(name="Write")],
        )
        session = make_session(messages=[make_message(Role.USER, "do it"), msg])
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        tool_items = [i for i in items if i.type == KnowledgeType.TOOL_PATTERN]
        assert len(tool_items) == 1
        assert "Read" in tool_items[0].content
        assert "Write" in tool_items[0].content

    def test_extracts_decision(self) -> None:
        msg = make_message(
            Role.ASSISTANT,
            "After reviewing options, we decided to use PostgreSQL for the database layer because of its JSON support.",
        )
        session = make_session(messages=[make_message(Role.USER, "what db?"), msg])
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        decisions = [i for i in items if i.type == KnowledgeType.DECISION]
        assert len(decisions) == 1

    def test_extracts_insight(self) -> None:
        msg = make_message(
            Role.ASSISTANT,
            "Importantly: the cache invalidation must happen before the write to avoid stale reads.",
        )
        session = make_session(messages=[make_message(Role.USER, "explain"), msg])
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        insights = [i for i in items if i.type == KnowledgeType.INSIGHT]
        assert len(insights) == 1

    def test_extracts_code_artifacts(self) -> None:
        code = "```python\ndef foo():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z\n```"
        msg = make_message(Role.ASSISTANT, f"Here is the code:\n{code}")
        session = make_session(messages=[make_message(Role.USER, "write code"), msg])
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        artifacts = [i for i in items if i.type == KnowledgeType.CODE_ARTIFACT]
        assert len(artifacts) == 1
        assert "python" in artifacts[0].tags

    def test_skips_short_code_blocks(self) -> None:
        code = "```python\nx = 1\n```"
        msg = make_message(Role.ASSISTANT, f"Short:\n{code}")
        session = make_session(messages=[make_message(Role.USER, "hi"), msg])
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        artifacts = [i for i in items if i.type == KnowledgeType.CODE_ARTIFACT]
        assert len(artifacts) == 0

    def test_knowledge_items_have_source_session(self) -> None:
        session = make_session(id="sess-42")
        normalizer = Normalizer()
        items = normalizer.normalize(session)
        for item in items:
            assert item.source_session == "sess-42"
