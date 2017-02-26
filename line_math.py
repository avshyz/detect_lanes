from collections import namedtuple
from typing import Sequence

Line = namedtuple("Line", "x1 y1 x2 y2")


def average_of_lines(lines: Sequence[Line]) -> Sequence[Line]:
    """ Returns the avg of a sequence of lines. Rounds float points down to the nearest int.
     Returns an empty list if there are no lines present."""

    n_lines = len(lines)
    if n_lines < 2:
        return lines

    avg_x1 = int(sum(line.x1 for line in lines) / n_lines)
    avg_y1 = int(sum(line.y1 for line in lines) / n_lines)
    avg_x2 = int(sum(line.x2 for line in lines) / n_lines)
    avg_y2 = int(sum(line.y2 for line in lines) / n_lines)

    return [Line(avg_x1, avg_y1, avg_x2, avg_y2)]


def intercept(line: Line) -> float:
    """ Returns the B for a line represented by `y = mx + b` """
    # y = mx + b
    # b = y - mx
    return line.y1 - slope(line) * line.x1


def extrapolate(line: Line, fit_y1: int, fit_y2: int) -> Line:
    line_slope = slope(line)
    b = intercept(line)

    # x = (y - b) / m
    x1 = (fit_y1 - b) / line_slope
    x2 = (fit_y2 - b) / line_slope
    return Line(int(x1), int(fit_y1), int(x2), int(fit_y2))


def slope(line: Line) -> float:
    """ Returns the slope of a line. """
    rise = (line.y2 - line.y1)
    run = (line.x2 - line.x1)
    # We're trading off a tiny bit of accuracy for our program not crashing.
    if run == 0:
        run = 0.000000001
    return rise / run
