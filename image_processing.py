from typing import Sequence

import cv2
import numpy as np

from line_math import Line


def grayscale(img: np.ndarray):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Applies a Gaussian Noise kernel. kernel_size: must be an odd int """
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


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    Merge two images together

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def make_lines_image(lines: Sequence[Line], height, width, color=(255, 0, 0), thickness=10):
    """
    Draw lines on a black image with size height and width.

    :param lines: Indicates positions to draw on return image
    :param height: of the return image
    :param width: of the return image
    :param color: of lines
    :param thickness: of lines
    """
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    for line in lines:
        cv2.line(line_img, (line.x1, line.y1), (line.x2, line.y2), color, thickness)
    return line_img
