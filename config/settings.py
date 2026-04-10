from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import os


@dataclass
class AgentConfig:
    """Configuration for a single AI agent source."""

    name: str
    enabled: bool = True
    source_path: Optional[Path] = None


@dataclass
class Settings:
    """Global pipeline configuration."""

    vault_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("VAULTPATH", os.path.expanduser("~/Obsidian"))
        )
    )
    state_dir: Path = field(default_factory=lambda: Path(".state"))

    agents: Dict[str, AgentConfig] = field(default_factory=lambda: {
        "claude-code": AgentConfig(
            name="claude-code",
            source_path=Path("~/.claude/projects").expanduser(),
        ),
        "codex": AgentConfig(
            name="codex",
            source_path=Path("~/.codex/sessions").expanduser(),
        ),
        "cursor": AgentConfig(
            name="cursor",
            source_path=Path("~/.cursor/chats").expanduser(),
        ),
        "factory-droid": AgentConfig(
            name="factory-droid",
            source_path=Path("~/.factory/sessions").expanduser(),
        ),
        "pi": AgentConfig(
            name="pi",
            source_path=Path("~/.centralagent/pi-import").expanduser(),
        ),
    })

    # Output paths (relative to vault_path)
    session_log_dir: str = "Sessions"
    extract_dir: str = "agents"

    # Processing options
    max_messages_per_session: int = 500
    skip_empty_sessions: bool = True
    include_assistant_responses: bool = True
    include_tool_usage: bool = True
