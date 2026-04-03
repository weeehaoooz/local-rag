"""Progress tracking utilities for indexing."""

from typing import Protocol, TypeVar, Generic
from rich.console import Console
from rich.text import Text

T = TypeVar('T')


class ProgressHandler(Protocol[T]):
    """Protocol for progress handlers."""
    def update(self, current: int, total: int, message: str) -> None:
        """Update progress."""
        ...

    def start(self, message: str) -> None:
        """Start tracking."""
        ...

    def end(self) -> None:
        """End tracking."""
        ...

    def clear(self) -> None:
        """Clear the current progress display."""
        ...


def create_progress_handler(use_rich: bool = True) -> ProgressHandler:
    """Create a progress handler based on the configured mode.

    Args:
        use_rich: If True, use Rich's Live display (default). If False,
                  return a dummy handler that doesn't display anything.

    Returns:
        A ProgressHandler implementation.
    """
    if use_rich:
        console = Console()
        return RichProgressHandler(console=console)
    else:
        return DummyProgressHandler()


class RichProgressHandler(Generic[T]):
    """Rich-based progress handler with live display."""

    def __init__(self, console: Console):
        self.console = console
        self.progress: list['RichProgress'] = []
        self.current_progress: RichProgress | None = None

    def _render(self) -> Text:
        """Render all active progress bars."""
        if not self.current_progress:
            return Text()

        lines: list[Text] = []

        for prog in self.progress:
            if prog.is_active:
                elapsed = prog.end_time - prog.start_time if prog.start_time else 0
                elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else "0s"
                prog_str = str(prog)
                lines.append(Text(f"[bold]{prog.title}[/bold] {prog_str} ({elapsed_str})") if prog_str else f"{prog.title}:")

        if self.current_progress and self.current_progress.is_active:
            lines.append(self.current_progress)

        if self.progress:
            summary = Text()
            for prog in self.progress:
                if prog.is_active:
                    summary.append(" | ")
                    summary.append(Text(f"[dim]{prog.title}[/dim]"))
            self.current_progress = summary

        if not self.current_progress:
            self.current_progress = Text()

        return self.current_progress

    def update(self, current: int, total: int, message: str) -> None:
        """Update a specific progress item."""
        if getattr(self.current_progress, "is_active", True) is False and self.progress:
            # Remove any finished progress bars
            self.progress = [p for p in self.progress if p.is_active]
            self.current_progress = None

        if message:
            title = message
        else:
            title = f"Task {current}/{total}" if total else "Task"

        self.current_progress = RichProgress(title=title, current=current, total=total)
        self.progress.append(self.current_progress)

    def start(self, message: str) -> None:
        """Start a new progress tracking."""
        pass

    def end(self) -> None:
        """End tracking and render final result."""
        if self.current_progress:
            self.current_progress.current = self.current_progress.total if self.current_progress.total else self.current_progress.current
            self.current_progress.message = "Done"
            self.progress.append(self.current_progress)
            self.console.print(self._render())
            self.current_progress = None

    def clear(self) -> None:
        """Clear the current progress display."""
        self.current_progress = None
        self.progress = [p for p in self.progress if not p.is_active]


class DummyProgressHandler(Generic[T]):
    """Dummy progress handler for headless environments."""

    def update(self, current: int, total: int, message: str) -> None:
        """Update progress (no-op)."""
        pass

    def start(self, message: str) -> None:
        """Start tracking (no-op)."""
        pass

    def end(self) -> None:
        """End tracking (no-op)."""
        pass

    def clear(self) -> None:
        """Clear display (no-op)."""
        pass


class RichProgress:
    """A single Rich progress bar."""

    def __init__(self, title: str, current: int, total: int = 0):
        self.title = title
        self.current = current
        self.total = total
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.is_active = True
        self.message: str | None = None

    @property
    def percent(self) -> float:
        """Calculate percentage."""
        if self.total <= 0:
            return 0
        return (self.current / self.total) * 100

    def __str__(self) -> str:
        if self.total == 0:
            return f"{self.percent:.1f}%"
        return f"{self.percent:.1f}%|{self.current}/{self.total}"
