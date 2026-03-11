"""Peach ASCII art banner — pure ANSI, no Rich dependency."""
from __future__ import annotations

# Peach color: #FF9B7A -> 256-color approx is 216, or use truecolor
_PEACH = "\033[1;38;2;255;155;122m"  # bold + truecolor peach
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_PEACH_TEXT = [
    "                                               ",
    "                                     ,,        ",
    "                                   `7MM        ",
    "                                     MM        ",
    '`7MMpdMAo.  .gP"Ya   ,6"Yb.  ,p6"bo  MMpMMMb.  ',
    "  MM   `Wb ,M'   Yb 8)   MM 6M'  OO  MM    MM  ",
    '  MM    M8 8M""""""  ,pm9MM 8M       MM    MM  ',
    "  MM   ,AP YM.    , 8M   MM YM.    , MM    MM  ",
    "  MMbmmd'   `Mbmmd' `Moo9^Yo.YMbmd'.JMML  JMML.",
    "  MM                                           ",
    ".JMML.                                         ",
]


def _box_lines(lines: list[str]) -> list[str]:
    """Wrap lines in a simple Unicode box border, colored peach."""
    max_w = max(len(l) for l in lines)
    top = f"{_PEACH}\u250c{'─' * (max_w + 2)}\u2510{_RESET}"
    bot = f"{_PEACH}\u2514{'─' * (max_w + 2)}\u2518{_RESET}"
    mid = []
    for l in lines:
        padded = l.ljust(max_w)
        mid.append(f"{_PEACH}\u2502{_RESET} {_PEACH}{padded}{_RESET} {_PEACH}\u2502{_RESET}")
    return [top] + mid + [bot]


def print_peach_art() -> None:
    """Print the peach art banner to stdout using ANSI escapes."""
    for line in _box_lines(_PEACH_TEXT):
        print(line)
