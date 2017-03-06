# Detect Lanes
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
Easily identify lane lines in driving data.

## Usage

#### Command Line:

    python detect_lanes.py <path_to_your_source_image> <path_to_output_destination>

#### In Code

    import detect_lanes


    # annotate image
    image = mpimg.imread(source)
    image_copy = np.copy(image)
    with_lane_highlights = detect_lanes.annotate_lanes(image_copy)
    write_image(dest, with_lane_highlights)


    # Annotate a Video
    from moviepy.editor import VideoFileClip
    white_output = 'test_videos_output/white.mp4'
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(detect_lanes.annotate_lanes) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


## Setup

#### Installing OpenCV

    >>> pip install pillow
    >>> conda install -c menpo opencv3=3.1.0

then to test if OpenCV is installed correctly:

    >>> python
    >>> import cv2
    >>> (i.e. did not get an ImportError)

(Ctrl-d to exit Python)

#### Installing moviepy

    >>> pip install moviepy`

and check that the install worked:

    >>> python
    >>> import moviepy
    >>> (i.e. did not get an ImportError)

