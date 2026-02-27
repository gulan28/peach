"""Peach ASCII art banner."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


_PEACH_TEXT = [
    "                                               ",
    "                                     ,,        ",
    "                                   `7MM        ",
    "                                     MM        ",
    "`7MMpdMAo.  .gP\"Ya   ,6\"Yb.  ,p6\"bo  MMpMMMb.  ",
    "  MM   `Wb ,M'   Yb 8)   MM 6M'  OO  MM    MM  ",
    "  MM    M8 8M\"\"\"\"\"\"  ,pm9MM 8M       MM    MM  ",
    "  MM   ,AP YM.    , 8M   MM YM.    , MM    MM  ",
    "  MMbmmd'   `Mbmmd' `Moo9^Yo.YMbmd'.JMML  JMML.",
    "  MM                                           ",
    ".JMML.                                         ",
]


def get_peach_art() -> Text:
    """Return the 'peach' FIGlet text in orangish-peach shade."""
    art = Text()
    for line in _PEACH_TEXT:
        art.append(line + "\n", style="bold #FF9B7A")
    return art


def print_peach_art(console: Console | None = None) -> None:
    """Print the peach art inside a box to the given console."""
    if console is None:
        console = Console()
    panel = Panel(
        get_peach_art(),
        border_style="#FF9B7A",
        expand=False,
    )
    console.print(panel)
