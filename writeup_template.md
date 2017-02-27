#**Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

* convert the image to grayscale, so it's easier to detect contrast without having to worry about different colors.
* Set pixels outside a specified region black: This allows us to use domain knowledge to detect only lines that will likely be related to lane markings.
* apply gaussian blur: to reduce noise the in the image, so we don't predict a bunch of tiny edges that don't exist.
* apply canny transform: This will look at the gradient of each pixel of the image (how different each pixel is to its neighbors).
  It will then select pixels with an intensity above a certain threshold to be edges. Pixels adjacent to high threshold pixels
  that are still above a lower threshold will be included as well. This converts the image to **line world!**
* Convert to hough space: try to draw lines between our images using the HoughLinePFunction
* heuristically remove lines that seem unlikely to be lane lines
  * too high up in the image
  * if we think the are the edged of our area of interest
* Split lines between left and right lanes and average each group.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline

My pipeline is fairly brittle. It will likely run into problems if:

  * We are on a very curved road
  * there are high contrast shadows on the road
  * We are already not in the center of the road
  * There are other types of markings on the road

One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...