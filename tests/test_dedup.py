"""Tests for the deduplication tracker."""

import json
from pathlib import Path

import pytest

from src.dedup import DedupTracker


class TestDedupTracker:
    """Tests for DedupTracker state management."""

    def test_initial_state_empty(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        assert not tracker.is_exported("claude-code", "session-1")

    def test_mark_exported_persists(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_exported("claude-code", "session-1")
        assert tracker.is_exported("claude-code", "session-1")

        # Reload from disk
        tracker2 = DedupTracker(state_dir=tmp_path)
        assert tracker2.is_exported("claude-code", "session-1")

    def test_mark_exported_no_duplicates(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_exported("claude-code", "session-1")
        tracker.mark_exported("claude-code", "session-1")
        entry = tracker._state["agents"]["claude-code"]
        assert entry["exported_sessions"].count("session-1") == 1

    def test_mark_batch_exported(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_batch_exported("codex", ["s1", "s2", "s3"])
        assert tracker.is_exported("codex", "s1")
        assert tracker.is_exported("codex", "s2")
        assert tracker.is_exported("codex", "s3")
        assert not tracker.is_exported("codex", "s4")

    def test_batch_no_duplicates(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_exported("codex", "s1")
        tracker.mark_batch_exported("codex", ["s1", "s2"])
        entry = tracker._state["agents"]["codex"]
        assert entry["exported_sessions"].count("s1") == 1

    def test_get_exported_sessions(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_batch_exported("claude-code", ["a", "b"])
        result = tracker.get_exported_sessions("claude-code")
        assert result == {"a", "b"}

    def test_get_last_run_none_initially(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        assert tracker.get_last_run("claude-code") is None

    def test_get_last_run_after_export(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_exported("claude-code", "s1")
        last_run = tracker.get_last_run("claude-code")
        assert last_run is not None
        assert "T" in last_run  # ISO format

    def test_reset_single_agent(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_exported("claude-code", "s1")
        tracker.mark_exported("codex", "s2")
        tracker.reset("claude-code")
        assert not tracker.is_exported("claude-code", "s1")
        assert tracker.is_exported("codex", "s2")

    def test_reset_all_agents(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_exported("claude-code", "s1")
        tracker.mark_exported("codex", "s2")
        tracker.reset()
        assert not tracker.is_exported("claude-code", "s1")
        assert not tracker.is_exported("codex", "s2")

    def test_agents_isolated(self, tmp_path: Path) -> None:
        tracker = DedupTracker(state_dir=tmp_path)
        tracker.mark_exported("claude-code", "session-1")
        assert not tracker.is_exported("codex", "session-1")

    def test_corrupted_state_file(self, tmp_path: Path) -> None:
        state_file = tmp_path / "last_run.json"
        state_file.write_text("not valid json", encoding="utf-8")
        tracker = DedupTracker(state_dir=tmp_path)
        assert not tracker.is_exported("claude-code", "s1")

    def test_state_dir_created_on_save(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "nested" / "state"
        tracker = DedupTracker(state_dir=state_dir)
        tracker.mark_exported("claude-code", "s1")
        assert state_dir.exists()
        assert (state_dir / "last_run.json").exists()
