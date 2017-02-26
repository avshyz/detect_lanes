import numpy as np

from image_processing import grayscale, region_of_interest, gaussian_blur, canny, hough_lines, weighted_img

rho = 2
theta = np.pi / 180
threshold = 100
min_line_len = 30
max_line_gap = 100


def process_image(image: np.ndarray) -> np.ndarray:
    first_img = np.copy(image)
    height, width, _ = first_img.shape

    gray = grayscale(first_img)

    top_vertex_y = int((2 / 5) * height)
    top_vertex_x = int(width / 2)
    vertices = np.array([(0, height), (top_vertex_x, top_vertex_y), (width, height)])
    region = region_of_interest(gray, [vertices])  # HMM, how do we get rid of the lines from our region selection

    blurred = gaussian_blur(region, 3)  # maybe we can filter all non lanes by raising blur
    canny_img = canny(blurred, 150, 300)
    cropped_canny = region_of_interest(canny_img, [np.array([(30, 540), (490, 260), (940, 530)])])

    lines = hough_lines(cropped_canny, rho, theta, threshold, min_line_len, max_line_gap)
    return weighted_img(lines, first_img)


