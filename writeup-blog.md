# Detecting Lane Lines in Python with OpenCV
I recently built a [lane detection tool]() for the first project of the [Self Driving Car Nano Degree](https://www.udacity.com/drive) program through Udacity.
The goal of the project was to detect lane lines on overhead camera video footage taken while a person was driving.

[//]: # (Image References)
[image1]: ./examples/grayscale.jpg "Grayscale"
[gray]: ./examples/gray.jpg "gray"
[region of interest]: ./examples/region_selected.jpg "hello"
[blurred]: ./examples/blurred.jpg
[canny]: ./examples/canny.jpg
[lines]: ./examples/lines.jpg
[final]: ./test_images/solidWhiteCurve_annotated.jpg
[compare]: ./self_driving_car_begin_end.png

Basically, a tool to turn the left video into the right one:

![Compare](https://media.giphy.com/media/3oKIPDOxc9XKGqh2O4/giphy.gif)

In this post we'll go over what learned for people who are interested in Computer Vision or people hoping to
get a glimpse into the Self Driving Car Nano Degree.
I'll follow this post up with a post on my general impressions on the program after the first two weeks.
It's worth noting, this implementation does not use any **state of the art machine learning techniques**
cough, cough, deep learning. We'll revisit the project from that angle, but it was awesome to see how much is possible with
edge detection and logic.

### Generating candidate edges and choosing our favorites.
The pipeline has two major phases: the first is prepping for and detecting lines in the image,
and the second is heuristically removing lines that don't seem like sensible candidates to represent lanes.

[image of pipeline]

* [Generate Candidate Lines](#generate-candidate-lines)
  * [Convert Image to Grayscale](#convert-image-to-grayscale)
  * [Choose a Region of Interest](#choose-a-region-of-interest)
  * [Apply Gaussian Blur](#apply-gaussian-blut)
  * [Apply the Canny Transform](#canny-transform)
  * [Detect Lines by Converting Image to Hough Space](#detect-lines)
* [Heuristically Process Lines](#heuristically-process-lines)

### Generate Candidate Lines

#### Convert Image to Grayscale
This makes it easier for us to detect contrast without having to deal with different colors.
We can just work with the brightness intensity of the image. (cv2 references opencv)

    grayscale_img = cv2.cvtColor(initial_image, cv2.COLOR_RGB2GRAY)

![Look at all those gray scales][gray]

#### Choose a Region Of Interest
Set pixels outside a specified region black: This allows us to use domain knowledge to detect only lines that will likely be related to lane markings.

First, we specify a region we want to focus on. Let's draw a triangle from the two bottom corners to the center of the image.

    height, length, _ = image.shape
    bottom_left_corner = (0, height)                                      # (1)
    bottom_right_corner = (length, height)
    center = (int(length / 2), int(height/2))
    region = [np.array([bottom_left_corner,center,bottom_right_corner])]  # (2)

(1) It's worth noting points are specified (x_pos, y_pos). `x_pos` goes from left to right and `y_pos` goes from top to bottom.
(2) You have to wrap your points in an np.array **AND** a second outer python `List`.
This was not well documented and seemed to be a large source of confusion for several people on the message boards.

Masking a grayscale image for a certain region looks like this:

    mask = np.zeros_like(grayscale_img)                            # (1)
    keep_region_color = 255
    cv2.fillPoly(mask, region, ignore_mask_color)                  # (2)
    region_selected_image = cv2.bitwise_and(grayscale_img, mask)   # (3)

(1) make a 2d array the same size as `grayscale_img` with all 0 values.
(2) set the area within specified regions to nonzero values.
(3) create an image keeping pixels from `grayscale_img` that correspond to nonzero pixels in mask.

![This is all we're interested in][region of interest]

#### Apply Gaussian Blur
We do this to reduce noise the in the image. We don't predict a bunch of tiny edges that don't mean much to us.

    kernel_size = 3
    blurred =  cv2.GaussianBlur(region_selected_image, (kernel_size, kernel_size), 0)

![Just a little blur][blurred]

#### Canny Transform
This will look at the gradient of each pixel of the image (how different each pixel is to its neighbors).
  It will then select pixels with an intensity above a certain threshold to be edges. Pixels adjacent to high threshold pixels
  that are still above a lower threshold will be included as well. This converts the image to **line world!**

    low_threshold = 150
    high_threshold = 300
    canny_transformed = cv2.Canny(blurred, low_threshold, high_threshold)

![After we apply the canny transform][canny]

#### Detect Lines
Convert our image to Hough Space. We try to draw lines along edges in our image using the `HoughLinePFunction`:

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
        maxLineGap=max_line_gap)

    Line = namedtuple("Line", "x1 y1 x2 y2")
    lines = [Line(*line[0]) for line in lines]         # (1)

(1) I made my own line data structure to make working with the lines friendlier

#### Heuristically Process Lines (`reduce_to_lanes`)
Now that we have all of our lines to work with. We can use logic to guess what we think are the lane lines. We'll do things like:
  * Remove lines that have too small a magnitude of slope. These lines will be too horizontal for us.
  * Remove lines that are too high up in the image. These are the sky or other non lane lines stuff.
  * Split lines into probable left and right lanes based on slope
  * Separately average all left and right lanes together

![Final Result Lines][lines]

Now we can make an image to represent our lines and glue it to our copy of the original image:

    lanes = _reduce_to_lanes(lines) # (1)

    color=(255, 0, 0)
    thickness=10
    lanes_img = np.zeros((height, width, 3), dtype=np.uint8)
    for lane in lines:
        cv2.line(lanes_img, (lane.x1, lane.y1), (lane.x2, lane.y2), color, thickness)

    lane_annotated = cv2.addWeighted(initial_img, .8, lanes_image, 1, 0)

(1) [See how I reduced my lines to just the lanes.]()

![final][final]

### Conclusion
We just stepped through a practical image processing pipeline using opencv and python! It goes without saying,
this is not production ready, but it's exciting to see how much we can do with a little image manipulating and line
filtering logic. What a nice starting point. Next, we'll talk about the experience of going through the first two weeks of the Udacity Self Driving Car Nanodegree.

#### Shortcomings

There are some shortcomings to this approach. As we mentioned at the top - this is not using any fancy machine learning.
The pipeline is fairly brittle. It will likely run into problems if:

* We are on a very curved road
* There are high contrast shadows on the road
* We are already not in the center of the road
* There are other types of markings on the road


#### Possible Improvements

* Keep a buffer of the lane lines detected in the last `n` frames and average your lane projection over your buffer.
This would smooth results and remove jumpiness. Also if you lose your lane for a frame it will handle it in a sensible way.
* Improve parameter tuning or calibrate parameters around certain features of the image.
Perhaps calibrate region of interest on previously seen lane lines. Try to use the horizon, etc.
* Train a ML classification model to predict if a line is a good lane candidate.
* Try to fit [curved lines]([curvy lanes](http://airccj.org/CSCP/vol5/csit53211.pdf)) for curved roads.
