from typing import Sequence, Tuple

import cv2
import numpy as np

from line_math import average_of_lines, extrapolate, slope, Line


def grayscale(img: np.ndarray):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Applies a Gaussian Noise kernel.
    :param img:
    :param kernel_size: must be an odd int """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img: np.ndarray, vertices: Sequence[np.array]) -> np.ndarray:
    """ Applies an image mask. Everything outside of the region defined by vertices will be set to black.
    Vertices should be in the form Sequence[np.array[Tuple[int, int]]] but we can't document this with mypy. """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    # Vertices is a list containing an np.array of vertices. This is not obvious.
    # It needs to look like: [np.array(vertex1, vertex2, vertex3)]
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap) -> Sequence[Line]:
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # Use our tuple object to make line calculations friendlier
    # A line is given to use in the format [[x1,y1,x2,y2]]
    return [Line(*line[0]) for line in lines]


def draw_lines(img, lines: Sequence[Line], color=(255, 0, 0), thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lines = reduce_to_lanes(lines)
    # lines = filter_too_horizontal(lines)
    height = img.shape[0]
    bottom = height
    # top = min(line.y1 for line in lanes) + 30
    top = 320

    for line in lines:
        extr = extrapolate(line, bottom, top)
        cv2.line(img, (extr.x1, extr.y1), (extr.x2, extr.y2), color, thickness)


def reduce_to_lanes(lines: Sequence[Line]) -> Sequence[Line]:
    # filter lines that are too high to be lane lines
    MIN_ACCEPTABLE_Y_POSITION = 400
    lines = [line for line in lines if line.y1 > MIN_ACCEPTABLE_Y_POSITION or line.y2 > MIN_ACCEPTABLE_Y_POSITION]

    # lines = filter_too_horizontal(lines)

    left_lanes = [line for line in lines if slope(line) > 0]
    right_lanes = [line for line in lines if slope(line) < 0]
    left_avg = average_of_lines(left_lanes)
    right_avg = average_of_lines(right_lanes)
    return left_avg + right_avg

def filter_too_horizontal(lines: Sequence[Line]) -> Sequence[Line]:
    slope_magnitudes = (np.abs(slope(line)) for line in lines)
    lane_slope = np.mean(reversed(sorted(slope_magnitudes))[:2])

    return [line for line in lines if np.abs(slope(line)) > 0.9*lane_slope]
    # pass


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def make_lines_image(lines: Sequence[Line], height, width):
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
