# Agent Guidance

CentralAgent Brain is a Python ingestion pipeline that imports AI-agent conversation sessions into an Obsidian vault. Favor small, well-tested changes that preserve parser boundaries and deterministic output.

## Project structure

- `src/cli.py` defines the `centralagent-brain` CLI and subcommands.
- `src/pipeline.py` orchestrates discovery, parsing, normalization, deduplication, and writing.
- `src/parsers/` contains one parser per agent source.
- `src/writers/` writes session logs and extracted knowledge into the vault.
- `src/models.py`, `src/normalizer.py`, and `src/dedup.py` hold shared domain logic.
- `config/settings.py` defines default source paths, output paths, and processing options.
- `tests/` contains parser, pipeline, normalizer, dedup, and writer tests with fixtures.

## Commands

- Install for local development: `python -m pip install -e ".[dev]"`
- Run the CLI from source: `python -m src status`, `python -m src list`, or `python -m src ingest --dry-run`
- Run tests: `pytest`
- Run a focused test: `pytest tests/test_pipeline.py`

## Environment and data

- `VAULTPATH` overrides the default Obsidian vault path of `~/Obsidian`.
- `.state/` stores deduplication state and should be treated as runtime state, not source.
- Default source paths include `~/.claude/projects`, `~/.codex/sessions`, `~/.cursor/chats`, `~/.factory/sessions`, and `~/.centralagent/pi-import`.
- Use `--dry-run` before changing parser or writer behavior that can affect user vault output.

## Implementation notes

- Keep parser-specific logic inside the matching file in `src/parsers/`.
- Normalize shared session semantics in `src/normalizer.py` rather than duplicating normalization in parsers.
- Add or update fixtures when parser behavior changes.
- Do not make broad formatting or refactor-only edits while fixing a parser bug.
- Preserve idempotency: dedup behavior should prevent duplicate exports unless `--force` is used.

## Validation

After changes, run the smallest relevant test first, then `pytest` for cross-parser changes. For ingestion changes, also run `python -m src ingest --dry-run --agent <agent-name>` when local fixtures or source data are available.
