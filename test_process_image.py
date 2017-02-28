import os
from typing import Sequence

import matplotlib.image as mpimg
import numpy as np

from detect_lanes import annotate_lanes, _find_lines
from pytest import fixture


@fixture()
def image_paths() -> Sequence[str]:
    images_dir = "./test_images"
    image_path_endings = os.listdir(images_dir)
    return [os.path.join(images_dir, ending) for ending in image_path_endings]


@fixture()
def image(image_paths: Sequence[str]) -> np.ndarray:
    first = image_paths[0]
    return mpimg.imread(first)


@fixture()
def images(image_paths: Sequence[str]) -> Sequence[np.ndarray]:
    return [mpimg.imread(path) for path in image_paths]


def test_can_process_image_without_crashing(image: np.ndarray):
    """ We don't have a good way of knowing if the image is being processed
    properly without eyeballing it. But at least with this function, we can
    check quickly if something crashes"""
    annotate_lanes(image)


def test_can_process_all_examples_without_crashing(images: Sequence[np.ndarray]):
    for image in images:
        annotate_lanes(image)



def test_has_lane_lines_for_third_image(images):
    """ Were not getting lane lines on the third image! Let's find out why. """

    third_image = images[2]
    lines = _find_lines(third_image)
    assert len(lines) > 0
