# Detecting Lane Lines in Python with OpenCV
I recently completed the first project of the [Self Driving Car Nano Degree](https://www.udacity.com/drive) program through Udacity.
The goal of the project was to detect lane lines on overhead camera video footage taken while a person was driving.

[image1]: ./examples/grayscale.jpg "Grayscale"
[gray]: ./examples/gray.jpg "gray"
[region of interest]: ./examples/region_selected.jpg "hello"
[blurred]: ./examples/blurred.jpg
[canny]: ./examples/canny.jpg
[lines]: ./examples/lines.jpg
[final]: ./test_images/solidWhiteCurve_annotated.jpg
[compare]: ./self_driving_car_begin_end.png

Basically, to turn the left image into the right image.

![compare][compare]

In this post we'll go over what learned for people who are interested in Computer Vision or people hoping to
get a glimpse into the Self Driving Car Nano Degree.
I'll follow this post up with a post on my general impressions on the program after the first two weeks.
It's worth noting, this implementation does not use any **state of the art machine learning techniques**
cough, cough, deep learning. We'll revisit the project from that angle, but it was awesome to see much is possible with
edge detection and logic.

### Generating candidate edges and choosing our favorites.
The pipeline has two major phases: the first is prepping for and detecting lines in the image,
and the second is heuristically removing lines that don't seem like sensible candidates to represent lanes.

[image of pipeline]

* Generate Candidate Lines
  * Convert Image to Grayscale
  * choose a region of interest
  * apply gaussian blur
  * apply the canny transform
  * detect lines by converting image to hough space

* Filtering / combining / processing lines to creating reasonable lane candidates


[//]: # (Image References)



### Generate Candidate Lines

#### Convert Image to Grayscale
This makes it easier for us to detect contrast without having to deal with different colors.
We can just work with the brightness intensity of the image.

        grayscale_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

![Look at all those gray scales][gray]

#### Choose a Region Of Interest
Set pixels outside a specified region black: This allows us to use domain knowledge to detect only lines that will likely be related to lane markings.

![This is all we're interested in][region of interest]

#### Apply Gaussian Blur
to reduce noise the in the image, so we don't predict a bunch of tiny edges that don't exist.

![Just a little blur][blurred]

#### Canny Transform
This will look at the gradient of each pixel of the image (how different each pixel is to its neighbors).
  It will then select pixels with an intensity above a certain threshold to be edges. Pixels adjacent to high threshold pixels
  that are still above a lower threshold will be included as well. This converts the image to **line world!**

![After we apply the canny transform][canny]

#### Detect Lines
* Convert to Hough Space: try to draw lines between our images using the HoughLinePFunction
* heuristically remove lines that seem unlikely to be lane lines

#### Heuristically process lines (`reduce_to_lanes`)
  * remove lines that have too small of a slope
  * remove lines that are too high up in the image
  * split lines into probable left and right lanes based on slope
  * separately average all left and right lanes together

![Final Result Lines][lines]

Now we can just glue our lines onto our original copy of the image:

![final][final]


###2. Shortcomings

The pipeline is fairly brittle. It will likely run into problems if:

  * We are on a very curved road
  * There are high contrast shadows on the road
  * We are already not in the center of the road
  * There are other types of markings on the road

One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Possible Improvements

* Keep a buffer of the lane lines detected in the last `n` frames and average your lane projection over your buffer.
This would smooth results and remove jumpiness. Also if you lose your lane for a frame it will handle it in a sensible way.
* Improve parameter tuning or calibrate parameters around certain features of the image.
Perhaps calibrate region of interest on previously seen lane lines. Try to use the horizon, etc.
* Train a ML classification model to predict if a line is a good lane candidate.
* Try to fit curved lines for curved roads.

