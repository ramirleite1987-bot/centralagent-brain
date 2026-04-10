"""Pipeline orchestrator for the CentralAgent Brain ingestion pipeline.

Chains: discover sessions -> parse each -> normalize -> deduplicate -> write output.
Handles per-agent and batch (--all) execution. Supports --dry-run mode.
Provides logging and progress reporting.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import Settings
from src.dedup import DedupTracker
from src.models import Session
from src.normalizer import Normalizer
from src.parsers import ParserRegistry
from src.parsers.base import BaseParser
from src.writers.extract_writer import ExtractWriter
from src.writers.session_writer import SessionWriter

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of pipeline execution for a single agent."""

    agent: str
    discovered: int = 0
    parsed: int = 0
    skipped_dedup: int = 0
    skipped_empty: int = 0
    skipped_error: int = 0
    written: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline run."""

    agents: Dict[str, AgentResult] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def total_discovered(self) -> int:
        return sum(r.discovered for r in self.agents.values())

    @property
    def total_written(self) -> int:
        return sum(r.written for r in self.agents.values())

    @property
    def total_errors(self) -> int:
        return sum(len(r.errors) for r in self.agents.values())


@dataclass
class SessionInfo:
    """Lightweight session info for listing."""

    session_id: str
    agent: str
    path: Path
    exported: bool = False


@dataclass
class AgentStatus:
    """Status info for an agent."""

    agent: str
    enabled: bool
    last_run: Optional[str]
    exported_count: int
    available_sessions: int


class Pipeline:
    """Orchestrates the full ingestion pipeline.

    Constructor takes Settings. Chains discover -> parse -> normalize ->
    deduplicate -> write for each agent.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self._dedup = DedupTracker(state_dir=self.settings.state_dir)
        self._normalizer = Normalizer()
        self._session_writer = SessionWriter()
        self._extract_writer = ExtractWriter()

    def _get_parser(self, agent_name: str) -> BaseParser:
        """Get parser instance for an agent."""
        parser_cls = ParserRegistry.get(agent_name)
        return parser_cls()

    def _resolve_agents(self, agents: Optional[List[str]] = None) -> List[str]:
        """Resolve which agents to process."""
        if agents:
            for name in agents:
                if name not in self.settings.agents:
                    raise ValueError(
                        f"Unknown agent '{name}'. "
                        f"Available: {list(self.settings.agents.keys())}"
                    )
            return agents
        return [
            name
            for name, cfg in self.settings.agents.items()
            if cfg.enabled
        ]

    def run(
        self,
        agents: Optional[List[str]] = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult:
        """Run the ingestion pipeline.

        Args:
            agents: List of agent names to process, or None for all enabled.
            dry_run: If True, parse and normalize but don't write files.
            force: If True, re-export already-processed sessions.

        Returns:
            PipelineResult with counts per agent.
        """
        result = PipelineResult(dry_run=dry_run)
        agent_names = self._resolve_agents(agents)

        for agent_name in agent_names:
            agent_result = self._run_agent(
                agent_name, dry_run=dry_run, force=force
            )
            result.agents[agent_name] = agent_result

        return result

    def _run_agent(
        self,
        agent_name: str,
        dry_run: bool = False,
        force: bool = False,
    ) -> AgentResult:
        """Run the pipeline for a single agent."""
        agent_result = AgentResult(agent=agent_name)

        # Get parser
        try:
            parser = self._get_parser(agent_name)
        except KeyError as exc:
            logger.warning("No parser for agent '%s': %s", agent_name, exc)
            agent_result.errors.append(str(exc))
            return agent_result

        # Discover sessions
        try:
            session_paths = parser.discover_sessions()
        except Exception as exc:
            logger.warning(
                "Failed to discover sessions for '%s': %s", agent_name, exc
            )
            agent_result.errors.append(f"Discovery failed: {exc}")
            return agent_result

        agent_result.discovered = len(session_paths)
        logger.info(
            "Agent '%s': discovered %d sessions", agent_name, len(session_paths)
        )

        if force:
            self._dedup.reset(agent_name)

        vault_path = self.settings.vault_path
        exported_ids: List[str] = []

        for path in session_paths:
            # Parse session
            try:
                session = parser.parse_session(path)
            except Exception as exc:
                logger.warning(
                    "Failed to parse session '%s': %s", path, exc
                )
                agent_result.skipped_error += 1
                agent_result.errors.append(f"Parse error ({path}): {exc}")
                continue

            agent_result.parsed += 1

            # Skip empty sessions
            if self.settings.skip_empty_sessions and not session.messages:
                logger.debug("Skipping empty session: %s", session.id)
                agent_result.skipped_empty += 1
                continue

            # Dedup check
            if not force and self._dedup.is_exported(agent_name, session.id):
                logger.debug("Skipping already exported: %s", session.id)
                agent_result.skipped_dedup += 1
                continue

            # Normalize
            knowledge_items = self._normalizer.normalize(session)

            # Write output (unless dry-run)
            if not dry_run:
                try:
                    self._session_writer.write_session(session, vault_path)
                    if knowledge_items:
                        self._extract_writer.write_extract(
                            session, knowledge_items, vault_path
                        )
                    exported_ids.append(session.id)
                except Exception as exc:
                    logger.warning(
                        "Failed to write session '%s': %s", session.id, exc
                    )
                    agent_result.skipped_error += 1
                    agent_result.errors.append(f"Write error ({session.id}): {exc}")
                    continue

            agent_result.written += 1

        # Mark exported in dedup tracker (batch save)
        if exported_ids and not dry_run:
            self._dedup.mark_batch_exported(agent_name, exported_ids)

        logger.info(
            "Agent '%s': parsed=%d, written=%d, skipped(dedup=%d, empty=%d, error=%d)",
            agent_name,
            agent_result.parsed,
            agent_result.written,
            agent_result.skipped_dedup,
            agent_result.skipped_empty,
            agent_result.skipped_error,
        )

        return agent_result

    def list_sessions(
        self, agent: Optional[str] = None
    ) -> List[SessionInfo]:
        """List discovered sessions, optionally filtered by agent.

        Args:
            agent: Agent name to filter by, or None for all.

        Returns:
            List of SessionInfo objects.
        """
        agent_names = self._resolve_agents([agent] if agent else None)
        sessions: List[SessionInfo] = []

        for agent_name in agent_names:
            try:
                parser = self._get_parser(agent_name)
                paths = parser.discover_sessions()
            except Exception as exc:
                logger.warning(
                    "Failed to list sessions for '%s': %s", agent_name, exc
                )
                continue

            for path in paths:
                session_id = path.stem
                exported = self._dedup.is_exported(agent_name, session_id)
                sessions.append(
                    SessionInfo(
                        session_id=session_id,
                        agent=agent_name,
                        path=path,
                        exported=exported,
                    )
                )

        return sessions

    def status(self) -> Dict[str, AgentStatus]:
        """Get status info for all configured agents.

        Returns:
            Dictionary mapping agent names to AgentStatus objects.
        """
        result: Dict[str, AgentStatus] = {}

        for agent_name, cfg in self.settings.agents.items():
            last_run = self._dedup.get_last_run(agent_name)
            exported = self._dedup.get_exported_sessions(agent_name)

            available = 0
            if cfg.enabled:
                try:
                    parser = self._get_parser(agent_name)
                    available = len(parser.discover_sessions())
                except Exception:
                    pass

            result[agent_name] = AgentStatus(
                agent=agent_name,
                enabled=cfg.enabled,
                last_run=last_run,
                exported_count=len(exported),
                available_sessions=available,
            )

        return result
