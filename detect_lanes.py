from typing import Sequence

import cv2
import numpy as np

from image_processing import grayscale, region_of_interest, gaussian_blur, canny, hough_lines, weighted_img, \
    make_lines_image
from line_math import Line, average_of_lines, slope, extrapolate

RHO = 2
THETA = np.pi / 180
HOUGH_LINE_THRESHOLD = 50
MIN_LINE_LEN = 120
MAX_LINE_GAP = 150
TOP_VERTEX_HEIGHT_PERCENT = (6 / 11)
TOP_VERTEX_WIDTH_PERCENT = (1 / 2)

GAUSSIAN_BLUR_KERNEL_SIZE = 3

LOW_CANNY_THRESHOLD = 200
HIGH_CANNY_THRESHOLD = 300

MIN_ACCEPTABLE_Y_POSITION = 400
MIN_SLOPE = .4

LANE_TOP_Y_POSITION = 320

EXAMPLES_DIR = "./examples/"


def process_image(image: np.ndarray) -> np.ndarray:
    first_img = np.copy(image)
    height, width, _ = first_img.shape

    lines = find_lines(first_img)
    lanes = reduce_to_lanes(lines)

    # draw from the bottom of the image to near the horizon
    lanes = [extrapolate(lane, height, LANE_TOP_Y_POSITION) for lane in lanes]

    lines_img = make_lines_image(lanes, height, width)
    # first_img = region_of_interest(first_img, [slightly_smaller_vertices])
    return weighted_img(lines_img, first_img, .9)


def find_lines(image: np.ndarray) -> Sequence[Line]:
    height, width, _ = image.shape
    gray = grayscale(image)
    # cv2.imwrite(EXAMPLES_DIR + "gray.jpg", gray)
    vertices = region_of_interest_vertices(height, width)
    region = region_of_interest(gray, [vertices])  # HMM, how do we get rid of the lines from our region selection
    # cv2.imwrite(EXAMPLES_DIR + "region_selected.jpg", region)
    blurred = gaussian_blur(region, GAUSSIAN_BLUR_KERNEL_SIZE)  # maybe we can filter all non lanes by raising blur
    # cv2.imwrite(EXAMPLES_DIR + "blurred.jpg", blurred)
    canny_img = canny(blurred, LOW_CANNY_THRESHOLD, HIGH_CANNY_THRESHOLD)
    # cv2.imwrite(EXAMPLES_DIR + "canny.jpg", canny_img)

    slightly_smaller_vertices = vertices_just_inside(vertices)
    # cv2.imwrite(EXAMPLES_DIR + "canny.jpg", slightly_smaller_vertices)
    cropped_canny = region_of_interest(canny_img, [slightly_smaller_vertices])

    return hough_lines(cropped_canny, RHO, THETA, HOUGH_LINE_THRESHOLD, MIN_LINE_LEN, MAX_LINE_GAP)


def reduce_to_lanes(lines: Sequence[Line]) -> Sequence[Line]:
    """ Heuristically reduce a list of lines to what we believe are lanes. """

    # filter lines that are too high to be lane lines
    lines = [line for line in lines if line.y1 > MIN_ACCEPTABLE_Y_POSITION or line.y2 > MIN_ACCEPTABLE_Y_POSITION]

    # Let's get rid of horizontal lines.
    lines = [line for line in lines if np.abs(slope(line)) > MIN_SLOPE]

    left_lanes = [line for line in lines if slope(line) > 0]
    right_lanes = [line for line in lines if slope(line) < 0]
    left_avg = average_of_lines(left_lanes)
    right_avg = average_of_lines(right_lanes)
    return left_avg + right_avg


def region_of_interest_vertices(height: int, width: int) -> np.array:
    """ Create a region of interest by connecting a specified point to the two bottom corners of the img """
    pinch = 10
    top_vertex_y = int(TOP_VERTEX_HEIGHT_PERCENT * height)
    top_vertex_x = int(TOP_VERTEX_WIDTH_PERCENT * width)
    return np.array([(pinch, height), (top_vertex_x, top_vertex_y), (width - pinch, height)])


def vertices_just_inside(region: np.array) -> np.array:
    """ Returns a region just inside a given region. Regions are represented by a 3 element np.array of tuples. """
    delta_pixels = 70
    bottom_left, top, bottom_right = region
    inner_bottom_left = (bottom_left[0] + delta_pixels, bottom_left[1] - int(delta_pixels / 10))
    inner_top = (top[0], top[1] + int(delta_pixels / 10))
    inner_bottom_right = (bottom_right[0] - delta_pixels, bottom_right[1] - int(delta_pixels / 10))
    return np.array([inner_bottom_left, inner_top, inner_bottom_right])


def write_image(path, img):
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img_out)
