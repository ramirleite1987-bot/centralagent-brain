"""Core data models for the CentralAgent Brain ingestion pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Role(Enum):
    """Message sender role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class KnowledgeType(Enum):
    """Type of extracted knowledge item."""

    DECISION = "decision"
    INSIGHT = "insight"
    SUMMARY = "summary"
    CODE_ARTIFACT = "code_artifact"
    TOOL_PATTERN = "tool_pattern"
    ERROR_RESOLUTION = "error_resolution"


@dataclass
class ToolUse:
    """A single tool invocation within a message."""

    name: str
    input: Optional[Dict[str, Any]] = None
    output: Optional[str] = None


@dataclass
class Message:
    """A single message in a conversation session."""

    role: Role
    content: str
    timestamp: Optional[datetime] = None
    tool_uses: List[ToolUse] = field(default_factory=list)


@dataclass
class Session:
    """A parsed conversation session from an AI agent."""

    id: str
    agent: str
    timestamp: Optional[datetime] = None
    project: Optional[str] = None
    cwd: Optional[str] = None
    model: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None


@dataclass
class KnowledgeItem:
    """An extracted knowledge artifact from a session."""

    type: KnowledgeType
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    wikilinks: List[str] = field(default_factory=list)
    source_session: Optional[str] = None
