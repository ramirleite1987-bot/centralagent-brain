"""Deduplication tracker for the CentralAgent Brain ingestion pipeline.

Tracks exported session IDs in .state/last_run.json to avoid re-processing
sessions that have already been ingested. Supports --force flag to re-export all.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set


class DedupTracker:
    """Tracks which sessions have been exported to prevent duplicate ingestion.

    State is persisted to a JSON file containing per-agent timestamps and
    session ID sets.
    """

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self.state_dir = state_dir or Path(".state")
        self.state_file = self.state_dir / "last_run.json"
        self._state: Dict = self._load()
        self._exported_cache: Dict[str, Set[str]] = {}

    def _load(self) -> Dict:
        """Load state from disk, returning empty state if file doesn't exist."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"agents": {}}

    def _save(self) -> None:
        """Persist current state to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self._state, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _ensure_agent(self, agent: str) -> Dict:
        """Ensure agent entry exists in state and return it."""
        agents = self._state.setdefault("agents", {})
        if agent not in agents:
            agents[agent] = {
                "last_run": None,
                "exported_sessions": [],
            }
        # Build lookup set on first access for O(1) checks
        cache = self._exported_cache.get(agent)
        if cache is None:
            self._exported_cache[agent] = set(agents[agent]["exported_sessions"])
        return agents[agent]

    def is_exported(self, agent: str, session_id: str) -> bool:
        """Check if a session has already been exported."""
        self._ensure_agent(agent)
        return session_id in self._exported_cache[agent]

    def get_exported_sessions(self, agent: str) -> Set[str]:
        """Return the set of exported session IDs for an agent."""
        self._ensure_agent(agent)
        return set(self._exported_cache[agent])

    def mark_exported(self, agent: str, session_id: str) -> None:
        """Mark a session as exported."""
        entry = self._ensure_agent(agent)
        if session_id not in self._exported_cache[agent]:
            entry["exported_sessions"].append(session_id)
            self._exported_cache[agent].add(session_id)
        entry["last_run"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def mark_batch_exported(self, agent: str, session_ids: list[str]) -> None:
        """Mark multiple sessions as exported in a single save."""
        entry = self._ensure_agent(agent)
        for sid in session_ids:
            if sid not in self._exported_cache[agent]:
                entry["exported_sessions"].append(sid)
                self._exported_cache[agent].add(sid)
        entry["last_run"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_last_run(self, agent: str) -> Optional[str]:
        """Return the ISO timestamp of the last run for an agent, or None."""
        entry = self._ensure_agent(agent)
        return entry.get("last_run")

    def reset(self, agent: Optional[str] = None) -> None:
        """Reset state for one agent or all agents (for --force flag)."""
        if agent:
            self._state.setdefault("agents", {})[agent] = {
                "last_run": None,
                "exported_sessions": [],
            }
            self._exported_cache.pop(agent, None)
        else:
            self._state = {"agents": {}}
            self._exported_cache.clear()
        self._save()
