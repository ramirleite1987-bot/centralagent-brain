"""Abstract base writer interface and writer registry."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type

from src.models import KnowledgeItem, Session


class BaseWriter(ABC):
    """Abstract base class for all output writers.

    Each writer must implement methods for writing session logs
    and knowledge extracts to the Obsidian vault.
    """

    @abstractmethod
    def write_session(self, session: Session, output_dir: Path) -> Path:
        """Write a session log to the output directory.

        Args:
            session: The normalized session to write.
            output_dir: Root directory of the Obsidian vault.

        Returns:
            Path to the written file.
        """

    @abstractmethod
    def write_extract(
        self,
        session: Session,
        items: List[KnowledgeItem],
        output_dir: Path,
    ) -> Path:
        """Write knowledge extracts from a session.

        Args:
            session: The source session.
            items: Knowledge items extracted from the session.
            output_dir: Root directory of the Obsidian vault.

        Returns:
            Path to the written file.
        """


class WriterRegistry:
    """Registry that maps writer names to their classes."""

    _writers: Dict[str, Type[BaseWriter]] = {}

    @classmethod
    def register(cls, writer_class: Type[BaseWriter]) -> Type[BaseWriter]:
        """Register a writer class. Can be used as a decorator.

        Args:
            writer_class: The writer class to register.

        Returns:
            The writer class (unchanged), enabling use as a decorator.
        """
        name = writer_class.__name__
        cls._writers[name] = writer_class
        return writer_class

    @classmethod
    def get(cls, name: str) -> Type[BaseWriter]:
        """Retrieve a writer class by name.

        Args:
            name: The writer class name.

        Returns:
            The registered writer class.

        Raises:
            KeyError: If no writer is registered for the given name.
        """
        if name not in cls._writers:
            raise KeyError(
                f"No writer registered as '{name}'. "
                f"Available: {list(cls._writers.keys())}"
            )
        return cls._writers[name]

    @classmethod
    def all(cls) -> Dict[str, Type[BaseWriter]]:
        """Return all registered writers.

        Returns:
            Dictionary mapping names to writer classes.
        """
        return dict(cls._writers)
