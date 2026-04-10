"""Normalization engine for the CentralAgent Brain ingestion pipeline.

Transforms parsed Session objects into enriched session content and
structured KnowledgeItem instances (decisions, insights, patterns,
tools used, code artifacts). Implements slugification per MEMORY.md
rules and wikilink generation.
"""

import re
import unicodedata
from datetime import datetime
from typing import List, Optional

from src.models import (
    KnowledgeItem,
    KnowledgeType,
    Message,
    Role,
    Session,
)


def slugify(text: str) -> str:
    """Convert text to kebab-case slug per MEMORY.md rules.

    - Lowercase
    - NFKD normalization to strip accents
    - Spaces to hyphens
    - Remove non-alphanumeric (except hyphens)
    - Collapse multiple hyphens
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9-]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def make_wikilink(text: str) -> str:
    """Create an Obsidian wikilink from text."""
    return f"[[{text}]]"


class Normalizer:
    """Transforms parsed Sessions into enriched content and KnowledgeItems."""

    def normalize(self, session: Session) -> List[KnowledgeItem]:
        """Extract structured knowledge items from a session.

        Returns a list of KnowledgeItem instances representing decisions,
        insights, tool patterns, code artifacts, and a session summary.
        """
        items: List[KnowledgeItem] = []

        # Always produce a summary
        summary = self._extract_summary(session)
        if summary:
            items.append(summary)

        # Extract tool patterns
        items.extend(self._extract_tool_patterns(session))

        # Extract code artifacts
        items.extend(self._extract_code_artifacts(session))

        # Extract decisions and insights from conversation
        items.extend(self._extract_decisions_and_insights(session))

        return items

    def session_wikilinks(self, session: Session) -> List[str]:
        """Generate wikilinks for a session based on project and agent."""
        links: List[str] = []
        if session.project:
            links.append(make_wikilink(f"projects/{slugify(session.project)}"))
        links.append(make_wikilink(f"agents/{session.agent}"))
        return links

    def session_tags(self, session: Session) -> List[str]:
        """Generate tags for a session."""
        tags = [session.agent, "session"]
        if session.project:
            tags.append(slugify(session.project))
        return tags

    def session_slug(self, session: Session) -> str:
        """Generate a filename slug for a session.

        Format: {date}-{project}-{topic}
        """
        date_str = self._format_date(session.timestamp)
        project = slugify(session.project) if session.project else "unknown"
        topic = self._infer_topic(session)
        return f"{date_str}-{project}-{topic}"

    def _format_date(self, ts: Optional[datetime]) -> str:
        """Format a datetime as YYYY-MM-DD, defaulting to today."""
        if ts:
            return ts.strftime("%Y-%m-%d")
        return datetime.now().strftime("%Y-%m-%d")

    def _infer_topic(self, session: Session) -> str:
        """Infer a topic slug from the first user message."""
        for msg in session.messages:
            if msg.role == Role.USER and msg.content.strip():
                # Take first ~60 chars of first user message as topic
                text = msg.content.strip()[:60]
                # Truncate at last word boundary
                if len(msg.content.strip()) > 60:
                    last_space = text.rfind(" ")
                    if last_space > 20:
                        text = text[:last_space]
                return slugify(text)
        return "session"

    def _extract_summary(self, session: Session) -> Optional[KnowledgeItem]:
        """Create a summary KnowledgeItem for the session."""
        user_msgs = [m for m in session.messages if m.role == Role.USER]
        assistant_msgs = [m for m in session.messages if m.role == Role.ASSISTANT]

        if not user_msgs:
            return None

        # Build summary content
        lines = []
        lines.append(f"Session with {session.agent} on {self._format_date(session.timestamp)}")
        if session.project:
            lines.append(f"Project: {session.project}")
        if session.model:
            lines.append(f"Model: {session.model}")
        lines.append(f"Messages: {len(user_msgs)} user, {len(assistant_msgs)} assistant")

        # Include first user message as context
        first_msg = user_msgs[0].content.strip()
        if len(first_msg) > 200:
            first_msg = first_msg[:200] + "..."
        lines.append(f"\nFirst message: {first_msg}")

        return KnowledgeItem(
            type=KnowledgeType.SUMMARY,
            title=f"Session summary - {session.agent} - {self._format_date(session.timestamp)}",
            content="\n".join(lines),
            tags=self.session_tags(session),
            wikilinks=self.session_wikilinks(session),
            source_session=session.id,
        )

    def _extract_tool_patterns(self, session: Session) -> List[KnowledgeItem]:
        """Extract unique tool usage patterns from the session."""
        tool_names: dict[str, int] = {}
        for msg in session.messages:
            for tool in msg.tool_uses:
                tool_names[tool.name] = tool_names.get(tool.name, 0) + 1

        if not tool_names:
            return []

        # Create a single tool_pattern item summarizing all tools used
        lines = [f"- **{name}**: used {count} time(s)" for name, count in sorted(tool_names.items())]
        content = "\n".join(lines)

        return [
            KnowledgeItem(
                type=KnowledgeType.TOOL_PATTERN,
                title=f"Tools used - {session.agent} - {self._format_date(session.timestamp)}",
                content=content,
                tags=[session.agent, "tools"],
                wikilinks=self.session_wikilinks(session),
                source_session=session.id,
            )
        ]

    def _extract_code_artifacts(self, session: Session) -> List[KnowledgeItem]:
        """Extract code blocks from assistant messages as artifacts."""
        items: List[KnowledgeItem] = []
        code_block_re = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

        for msg in session.messages:
            if msg.role != Role.ASSISTANT:
                continue
            matches = code_block_re.findall(msg.content)
            for lang, code in matches:
                # Only extract substantial code blocks (>3 lines)
                if code.strip().count("\n") < 3:
                    continue
                title = f"Code artifact ({lang or 'text'}) - {session.agent}"
                # Truncate very large code blocks
                truncated = code.strip()
                if len(truncated) > 2000:
                    truncated = truncated[:2000] + "\n... (truncated)"
                items.append(
                    KnowledgeItem(
                        type=KnowledgeType.CODE_ARTIFACT,
                        title=title,
                        content=f"```{lang}\n{truncated}\n```",
                        tags=[session.agent, "code", lang] if lang else [session.agent, "code"],
                        wikilinks=self.session_wikilinks(session),
                        source_session=session.id,
                    )
                )
        return items

    def _extract_decisions_and_insights(self, session: Session) -> List[KnowledgeItem]:
        """Extract decisions and insights from assistant messages.

        Looks for signal phrases indicating decisions or insights.
        """
        items: List[KnowledgeItem] = []
        decision_patterns = re.compile(
            r"(?:decided to|decision:|choosing|we(?:'ll| will) go with|approach:)",
            re.IGNORECASE,
        )
        insight_patterns = re.compile(
            r"(?:important(?:ly)?:|note:|insight:|learned that|key takeaway|turns out)",
            re.IGNORECASE,
        )

        for msg in session.messages:
            if msg.role != Role.ASSISTANT:
                continue

            # Check for decisions
            for match in decision_patterns.finditer(msg.content):
                # Extract surrounding context (the sentence containing the match)
                start = max(0, msg.content.rfind("\n", 0, match.start()) + 1)
                end = msg.content.find("\n", match.end())
                if end == -1:
                    end = min(len(msg.content), match.end() + 200)
                sentence = msg.content[start:end].strip()
                if len(sentence) > 20:
                    items.append(
                        KnowledgeItem(
                            type=KnowledgeType.DECISION,
                            title=f"Decision - {session.agent} - {self._format_date(session.timestamp)}",
                            content=sentence,
                            tags=[session.agent, "decision"],
                            wikilinks=self.session_wikilinks(session),
                            source_session=session.id,
                        )
                    )
                    break  # One decision per session to avoid noise

            # Check for insights
            for match in insight_patterns.finditer(msg.content):
                start = max(0, msg.content.rfind("\n", 0, match.start()) + 1)
                end = msg.content.find("\n", match.end())
                if end == -1:
                    end = min(len(msg.content), match.end() + 200)
                sentence = msg.content[start:end].strip()
                if len(sentence) > 20:
                    items.append(
                        KnowledgeItem(
                            type=KnowledgeType.INSIGHT,
                            title=f"Insight - {session.agent} - {self._format_date(session.timestamp)}",
                            content=sentence,
                            tags=[session.agent, "insight"],
                            wikilinks=self.session_wikilinks(session),
                            source_session=session.id,
                        )
                    )
                    break  # One insight per session to avoid noise

        return items
