from line_math import average_of_lines, Line, slope, extrapolate


def test_average_of_lines():
    line_one = Line(0, 0, 2, 2)
    line_two = Line(2, 2, 4, 4)

    expected_avg = Line(1, 1, 3, 3)
    assert expected_avg == average_of_lines([line_one, line_two])


def test_average_of_lines_rounds_down():
    line_one = Line(0, 0, 1, 1)
    line_two = Line(2, 2, 4, 4)

    # if it didn't round. The avg would be (1,1,2.5,2.5)
    expected_avg = Line(1, 1, 2, 2)
    assert expected_avg == average_of_lines([line_one, line_two])


def test_slope():
    line = Line(0, 0, 1, 1)
    assert slope(line) == 1


def test_slope_handles_zero_run():
    line = Line(0, 0, 0, 1)
    assert slope(line) > 999999


def test_extrapolate():
    line = Line(1, 1, 2, 2)
    expected = Line(0, 0, 3, 3)
    assert expected == extrapolate(line, 0, 3)
