#**Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[gray]: ./examples/gray.jpg "gray"
[region of interest]: ./examples/region_selected.jpg "hello"
[blurred]: ./examples/blurred.jpg
[canny]: ./examples/canny.jpg
[lines]: ./examples/lines.jpg
[final]: ./test_images/solidWhiteCurve_annotated.jpg

---

## Reflection

### The Pipeline
The pipeline has two major phases: the first is prepping for and detecting lines in the image,
and the second is heuristically removing lines that don't seem like sensible candidates to represent lanes.

### Finding Candidate Lines

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

