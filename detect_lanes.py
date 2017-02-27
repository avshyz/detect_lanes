from typing import Sequence

import numpy as np
import cv2

from image_processing import grayscale, region_of_interest, gaussian_blur, canny, hough_lines, weighted_img, \
    make_lines_image
from line_math import Line

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

EXAMPLES_DIR = "./examples/"

def process_image(image: np.ndarray) -> np.ndarray:
    first_img = np.copy(image)
    height, width, _ = first_img.shape

    lines = find_lines(first_img)
    lines_img = make_lines_image(lines, height, width)
    # write_image(EXAMPLES_DIR + "lines.jpg", lines_img)

    # Just for testing so we can see where the region of interest is.
    # vertices = region_of_interest_vertices(height, width)
    # slightly_smaller_vertices = vertices_just_inside(vertices)
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
