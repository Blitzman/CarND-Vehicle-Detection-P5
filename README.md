# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training happens between lines 113 and 166 in `pipeline.py`. This process can be divided into four steps: feature extraction, data preparation and normalization, SVM training, and model loading/saving.

First, we load all car and non-car images and extract the aforementioned features for each one of them. We load a total of 8792 car images and 8968 non-car ones so the dataset is pretty well balanced. Those images are provided by Udacity for this project. Here are some examples of those images:

#TODO

Next step is data preparation. In this part we stack the features from car and non-car images for training, then a `StandardScaler` is used to normalize and scale those features. Labels are also generated and the function `train_test_split` is used to create training and testing splits.

After that, a `LinearSVC` is trained using the training partition and its accuracy is calculated on both training and testing sets obtaining a 100% precision in training and 98.37% in the test set.

At last, both the `LinearSVC` and the `StandardScaler` are saved to disk using `pickle` in order to avoid repeating this step for each execution. In subsequent runs of the application, both of them are just loaded from disk if one does not want to recompute features and train the SVM again.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The sliding window search process is codified into different parts of the program: a function to generate window candidates, a loop to use that function to generate multi-scale windows, and feature extraction and prediction for each one of the candidate windows.

The function `slide_window` is the candidate generation method. It takes an image, starting position for the X and Y axes, the window size, and the overlap amount. This function can be found in lines 172 through 208.

The `pipeline` function, which codifies the frame-by-frame vehicle detection process, makes use of that function in lines 259 to 262 to generate possible candidates. We use five different scales or window sizes: 48, 64, 96, 128, and 192 pixels with increasing overlaps 0.25, 0.25, 0.5, 0.85, and 0.85 respectively. Starting and ending points are set to avoid parts of the image in which no car will appear at all. This parameter setup decision was completely empirical. The candidate generation process produces the following result.

#TODO

Next, those windows are processed one by one in the main loop of the pipeline (lines 266 to 275). Each window is resized to `64x64` pixels and features are extracted for them using the aforementioned method. Then the trained SVM is used to predict whether or not they are cars. Detected windows are kept for subsequent steps.

#### 2. Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?

Our pipeline is mainly purposed for video processing since, as we will explain later, it relies on heat map averaging through frames to confidently detect cars. However, here we show the resulting test images disabling that video heat map averaging:

#TODO

The main feature introduced for classifier performance boosting, in terms of runtime, is the selection of proper window candidates that span only regions of the image where cars are expected to be, as well as a well defined set of scales to offer a good trade-off between detection and speed.

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to filter false positives and combine overlapping bounding boxes, we implemented video-based heatmap procedure. For the first frame, we build an empty heatmap and use the detected windows to "add heat" by adding a constant `heat_addition` to each pixel where a car was detected, then that value is multiplied by a `heat_multiplier` constant. By doing this we greatly boost those areas in which we already had a detection or heat.

After that, we threshold the heatmap to keep those areas in which a certain `heat_threshold` is met. That thresholded heatmap is passed to the `label` function to generate the final bounding boxes for the detections.

When the next frame is processed, we start with the previous heatmap and apply a decay constant `heat_decay` to it by multiplying each pixel by that constant. This decay will basically null each pixel in which a detection is not consistent through multiple frames. Then the heat addition and thresholding is repeated.

The functions for heat addition, decay, and thresholding can be found in lines 221 to 233. Those functions are used in the pipeline loop (lines 284 to 298) in `pipeline.py`.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

* Although the propagation of the heatmap helps making the detections somewhat consistent, the bounding boxes are not really stable. One possible improvement could be to not only propagate the heatmap but also average the bounding boxes through various frames.
* For the algorithm, it is impossible to distinguish when two or more cars are overlapping, detecting them as a single blob in the heatmap. This could be improved by keeping track of each car's bounding box an trying to force a separation in the bounding boxes when to cars are close.
* The pipeline is extremely inefficient, mainly due to the high amount of window candidates that must be checked for each frame. A better strategy for producing candidates could be devised, maybe trading accuracy for faster execution.
