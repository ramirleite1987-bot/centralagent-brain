"""Abstract base parser interface and parser registry."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type

from src.models import Session


class BaseParser(ABC):
    """Abstract base class for all agent parsers.

    Each parser must implement session discovery and parsing for a specific
    AI agent's conversation format.
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the canonical name of the agent this parser handles."""

    @abstractmethod
    def discover_sessions(self) -> List[Path]:
        """Discover all available session files for this agent.

        Returns:
            List of paths to session files that can be parsed.
        """

    @abstractmethod
    def parse_session(self, path: Path) -> Session:
        """Parse a single session file into a normalized Session object.

        Args:
            path: Path to the session file to parse.

        Returns:
            A normalized Session object.

        Raises:
            ValueError: If the session file cannot be parsed.
        """


class ParserRegistry:
    """Registry that maps agent names to their parser classes."""

    _parsers: Dict[str, Type[BaseParser]] = {}

    @classmethod
    def register(cls, parser_class: Type[BaseParser]) -> Type[BaseParser]:
        """Register a parser class. Can be used as a decorator.

        Args:
            parser_class: The parser class to register.

        Returns:
            The parser class (unchanged), enabling use as a decorator.
        """
        name = parser_class.agent_name.fget(None)  # type: ignore[attr-defined]
        cls._parsers[name] = parser_class
        return parser_class

    @classmethod
    def get(cls, agent_name: str) -> Type[BaseParser]:
        """Retrieve a parser class by agent name.

        Args:
            agent_name: The canonical agent name.

        Returns:
            The registered parser class.

        Raises:
            KeyError: If no parser is registered for the given agent name.
        """
        if agent_name not in cls._parsers:
            raise KeyError(
                f"No parser registered for agent '{agent_name}'. "
                f"Available: {list(cls._parsers.keys())}"
            )
        return cls._parsers[agent_name]

    @classmethod
    def all(cls) -> Dict[str, Type[BaseParser]]:
        """Return all registered parsers.

        Returns:
            Dictionary mapping agent names to parser classes.
        """
        return dict(cls._parsers)
