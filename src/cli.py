"""CLI argument parsing and dispatch for the CentralAgent Brain pipeline."""

import argparse
import logging
import sys
from typing import List, Optional

from src.pipeline import Pipeline

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with ingest, list, and status subcommands."""
    parser = argparse.ArgumentParser(
        prog="centralagent-brain",
        description="CentralAgent Brain ingestion pipeline — imports AI agent sessions into Obsidian vault.",
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
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Handle the ingest subcommand."""
    agents: Optional[List[str]] = None
    if args.agent:
        agents = [args.agent]
    # If neither --agent nor --all, process all enabled agents by default.

    pipeline = Pipeline()
    result = pipeline.run(
        agents=agents,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
    )

    if result.dry_run:
        print("[dry-run] No files were written.")

    print(
        f"Done: {result.total_discovered} discovered, "
        f"{result.total_written} written, "
        f"{result.total_errors} errors."
    )

    return 1 if result.total_errors > 0 else 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Handle the list subcommand."""
    pipeline = Pipeline()
    sessions = pipeline.list_sessions(agent=args.agent)

    if not sessions:
        print("No sessions found.")
        return 0

    for s in sessions:
        status = "exported" if s.exported else "new"
        print(f"[{s.agent}] {s.session_id} ({status}) — {s.path}")

    print(f"\nTotal: {len(sessions)} session(s)")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Handle the status subcommand."""
    pipeline = Pipeline()
    statuses = pipeline.status()

    if not statuses:
        print("No agents configured.")
        return 0

    for name, info in statuses.items():
        enabled_str = "enabled" if info.enabled else "disabled"
        last_run_str = info.last_run or "never"
        print(
            f"{name} ({enabled_str}): "
            f"last_run={last_run_str}, "
            f"exported={info.exported_count}, "
            f"available={info.available_sessions}"
        )

    return 0


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _configure_logging(getattr(args, "verbose", False))

    handlers = {
        "ingest": _cmd_ingest,
        "list": _cmd_list,
        "status": _cmd_status,
    }

    handler = handlers[args.command]
    try:
        exit_code = handler(args)
    except Exception as exc:
        logger.error("Command failed: %s", exc)
        sys.exit(1)

    sys.exit(exit_code)
