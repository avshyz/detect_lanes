import argparse
from typing import Sequence

import cv2
import matplotlib.image as mpimg
import numpy as np
import os

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

PIPELINE_EXAMPLES = "./pipeline_examples/"


def main():
    parser = argparse.ArgumentParser(description="annotate lane lines on an image of a road.")
    parser.add_argument("src_path", help="The path of the source image you would like annotated.")
    parser.add_argument("dest_path", help="where you'd like us to leave your nice annotated image.")
    args = parser.parse_args()
    annotate_lanes_file(args.src_path, args.dest_path)
    print("Annotated file:", args.src_path, "and left it at:", args.dest_path, "Drive safe!")


def annotate_lanes_batch(source_dir: str, dest_dir: str) -> None:
    """ Annotate a directory filled with images """
    image_path_endings = os.listdir(source_dir)
    source_paths = (os.path.join(source_dir, ending) for ending in image_path_endings)
    dest_paths = (os.path.join(dest_dir, ending) for ending in image_path_endings)

    for (source, dest) in zip(source_paths, dest_paths):
        annotate_lanes_file(source, dest)


def annotate_lanes_file(source_file_path: str, dest_file_path: str) -> None:
    image = mpimg.imread(source_file_path)
    image_copy = np.copy(image)
    with_lane_highlights = annotate_lanes(image_copy)
    # cv2.imwrite(dest_file_path, with_lane_highlights)
    _write_color_image(dest_file_path, with_lane_highlights)


def annotate_lanes(image: np.ndarray) -> np.ndarray:
    """ Returns a copy of the input image with lanes annotated. """
    first_img = np.copy(image)
    height, width, _ = first_img.shape

    lines = _find_lines(first_img)
    lanes = _reduce_to_lanes(lines)

    # draw from the bottom of the image to near the horizon
    lanes = [extrapolate(lane, height, LANE_TOP_Y_POSITION) for lane in lanes]

    lines_img = make_lines_image(lanes, height, width)
    # first_img = region_of_interest(first_img, [slightly_smaller_vertices])
    return weighted_img(lines_img, first_img, .9)


def find_lanes(image: np.ndarray) -> Sequence[Line]:
    """ Returns a list of Line objects corresponding to lanes. """
    first_img = np.copy(image)
    height, width, _ = first_img.shape

    lines = _find_lines(first_img)
    lanes = _reduce_to_lanes(lines)

    # draw from the bottom of the image to near the horizon
    return [extrapolate(lane, height, LANE_TOP_Y_POSITION) for lane in lanes]


def _find_lines(image: np.ndarray) -> Sequence[Line]:
    height, width, _ = image.shape
    gray = grayscale(image)
    vertices = _region_of_interest_vertices(height, width)
    region = region_of_interest(gray, [vertices])  # HMM, how do we get rid of the lines from our region selection
    blurred = gaussian_blur(region, GAUSSIAN_BLUR_KERNEL_SIZE)  # maybe we can filter all non lanes by raising blur
    canny_img = canny(blurred, LOW_CANNY_THRESHOLD, HIGH_CANNY_THRESHOLD)

    slightly_smaller_vertices = _vertices_just_inside(vertices)
    cropped_canny = region_of_interest(canny_img, [slightly_smaller_vertices])

    return hough_lines(cropped_canny, RHO, THETA, HOUGH_LINE_THRESHOLD, MIN_LINE_LEN, MAX_LINE_GAP)


def _reduce_to_lanes(lines: Sequence[Line]) -> Sequence[Line]:
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


def _region_of_interest_vertices(height: int, width: int) -> np.array:
    """ Create a region of interest by connecting a specified point to the two bottom corners of the img """
    pinch = 10
    top_vertex_y = int(TOP_VERTEX_HEIGHT_PERCENT * height)
    top_vertex_x = int(TOP_VERTEX_WIDTH_PERCENT * width)
    return np.array([(pinch, height), (top_vertex_x, top_vertex_y), (width - pinch, height)])


def _vertices_just_inside(region: np.array) -> np.array:
    """ Returns a region just inside a given region. Regions are represented by a 3 element np.array of tuples. """
    delta_pixels = 70
    bottom_left, top, bottom_right = region
    inner_bottom_left = (bottom_left[0] + delta_pixels, bottom_left[1] - int(delta_pixels / 10))
    inner_top = (top[0], top[1] + int(delta_pixels / 10))
    inner_bottom_right = (bottom_right[0] - delta_pixels, bottom_right[1] - int(delta_pixels / 10))
    return np.array([inner_bottom_left, inner_top, inner_bottom_right])

def _write_color_image(path, img):
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    cv2.imwrite(path, img_out)


if __name__ == '__main__':
    main()
