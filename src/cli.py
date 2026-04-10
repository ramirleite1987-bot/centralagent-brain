"""CLI argument parsing and dispatch for the CentralAgent Brain pipeline."""

import argparse
import logging
import sys
from datetime import datetime
from typing import List, Optional

from src.pipeline import Pipeline

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# ANSI helpers (auto-disabled when stdout is not a terminal)
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _COLOR else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _COLOR else text


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _COLOR else text


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if _COLOR else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _COLOR else text


def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m" if _COLOR else text


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _human_timestamp(iso_str: Optional[str]) -> str:
    """Convert ISO timestamp to human-friendly relative or absolute string."""
    if not iso_str:
        return _dim("never")
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now(dt.tzinfo)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            mins = seconds // 60
            return f"{mins}m ago"
        if seconds < 86400:
            hours = seconds // 3600
            return f"{hours}h ago"
        days = seconds // 86400
        if days == 1:
            return "yesterday"
        if days < 30:
            return f"{days}d ago"
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return iso_str


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with ingest, list, and status subcommands."""
    parser = argparse.ArgumentParser(
        prog="centralagent-brain",
        description="CentralAgent Brain — imports AI agent sessions into your Obsidian vault.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest ---
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run the ingestion pipeline to import sessions.",
    )
    ingest_parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Process only this agent (e.g. claude-code, cursor).",
    )
    ingest_parser.add_argument(
        "--all",
        dest="all_agents",
        action="store_true",
        default=False,
        help="Process all enabled agents.",
    )
    ingest_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Parse and normalize but do not write any files.",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-export already-processed sessions (ignore dedup).",
    )
    ingest_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging.",
    )

    # --- list ---
    list_parser = subparsers.add_parser(
        "list",
        help="List discovered sessions.",
    )
    list_parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Filter sessions by agent name.",
    )
    list_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging.",
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status",
        help="Show last run info and agent status.",
    )
    status_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging.",
    )

    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure logging level based on verbose flag."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(levelname)s: %(message)s"
    if verbose:
        fmt = "%(levelname)s [%(name)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_ingest(args: argparse.Namespace) -> int:
    """Handle the ingest subcommand."""
    agents: Optional[List[str]] = None
    if args.agent:
        agents = [args.agent]

    pipeline = Pipeline()

    if args.dry_run:
        print(_yellow("~ dry-run mode — no files will be written"))
        print()

    result = pipeline.run(
        agents=agents,
        dry_run=args.dry_run,
        force=args.force,
    )

    # Per-agent summary
    for agent_name, ar in result.agents.items():
        marker = _green("OK") if not ar.errors else _red("ERR")
        parts = [f"discovered={ar.discovered}"]
        if ar.parsed:
            parts.append(f"parsed={ar.parsed}")
        if ar.written:
            parts.append(f"written={_green(str(ar.written))}")
        if ar.skipped_dedup:
            parts.append(f"skipped(dedup)={ar.skipped_dedup}")
        if ar.skipped_empty:
            parts.append(f"skipped(empty)={ar.skipped_empty}")
        if ar.errors:
            parts.append(f"errors={_red(str(len(ar.errors)))}")
        print(f"  [{marker}] {_bold(agent_name):20s} {', '.join(parts)}")

    # Totals
    print()
    total_line = (
        f"  {_bold('Total')}: "
        f"{result.total_discovered} discovered, "
        f"{_green(str(result.total_written))} written, "
        f"{_red(str(result.total_errors)) if result.total_errors else '0'} errors"
    )
    print(total_line)

    if result.dry_run:
        print()
        print(_yellow("  ~ dry-run complete — no files were written"))

    return 1 if result.total_errors > 0 else 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Handle the list subcommand."""
    pipeline = Pipeline()
    sessions = pipeline.list_sessions(agent=args.agent)

    if not sessions:
        print(_dim("  No sessions found."))
        return 0

    # Group by agent
    by_agent: dict[str, list] = {}
    for s in sessions:
        by_agent.setdefault(s.agent, []).append(s)

    for agent_name, agent_sessions in sorted(by_agent.items()):
        new_count = sum(1 for s in agent_sessions if not s.exported)
        exported_count = len(agent_sessions) - new_count
        header = (
            f"  {_bold(agent_name)} "
            f"({_green(str(new_count))} new, "
            f"{_dim(str(exported_count))} exported)"
        )
        print(header)
        for s in agent_sessions:
            status = _dim("exported") if s.exported else _green("new     ")
            print(f"    {status}  {s.session_id}  {_dim(str(s.path))}")
        print()

    total_new = sum(1 for s in sessions if not s.exported)
    print(f"  {_bold('Total')}: {len(sessions)} session(s), {_green(str(total_new))} new")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Handle the status subcommand."""
    pipeline = Pipeline()
    statuses = pipeline.status()

    if not statuses:
        print(_dim("  No agents configured."))
        return 0

    # Header
    print(f"  {'Agent':<18s} {'Status':<10s} {'Last Run':<14s} {'Exported':>8s}  {'Available':>9s}")
    print(f"  {'─' * 18} {'─' * 10} {'─' * 14} {'─' * 8}  {'─' * 9}")

    for name, info in statuses.items():
        enabled_str = _green("enabled") if info.enabled else _red("disabled")
        last_run_str = _human_timestamp(info.last_run)
        exported_str = str(info.exported_count)
        avail_str = str(info.available_sessions)
        if info.available_sessions > 0 and info.available_sessions > info.exported_count:
            avail_str = _yellow(avail_str)
        print(
            f"  {_bold(name):<18s} {enabled_str:<10s} {last_run_str:<14s} "
            f"{exported_str:>8s}  {avail_str:>9s}"
        )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        print(f"{_bold('centralagent-brain')} v{VERSION}")
        print()
        print("  Import AI agent conversations into your Obsidian vault.")
        print()
        print(f"  {_bold('Commands')}:")
        print(f"    ingest   Run the ingestion pipeline")
        print(f"    list     List discovered sessions")
        print(f"    status   Show agent status and last run info")
        print()
        print(f"  Run {_cyan('centralagent-brain <command> --help')} for details.")
        sys.exit(0)

    _configure_logging(getattr(args, "verbose", False))

    handlers = {
        "ingest": _cmd_ingest,
        "list": _cmd_list,
        "status": _cmd_status,
    }

    print()
    handler = handlers[args.command]
    try:
        exit_code = handler(args)
    except Exception as exc:
        print(f"  {_red('Error')}: {exc}")
        logger.debug("Command failed", exc_info=True)
        sys.exit(1)

    print()
    sys.exit(exit_code)
