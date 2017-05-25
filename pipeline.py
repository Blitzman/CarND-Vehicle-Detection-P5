import cv2
import glob
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

import pickle

from skimage.feature import hog

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from moviepy.editor import VideoFileClip

from scipy.ndimage.measurements import label

###################################################################################################
## Load data for cars and not cars
###################################################################################################

cars = glob.glob('data/vehicles/**/*.png', recursive=True)
not_cars = glob.glob('data/non-vehicles/**/*.png', recursive=True)

print(str(len(cars)) + ' car images...')
print(str(len(not_cars)) + ' not car images...')

###################################################################################################
## Extract features
###################################################################################################

car_features = []
not_car_features = []

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel() 
    return features

def color_histogram(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def extract_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):

    features = None

    if hog_channel == "ALL":

        features = []

        for channel in range(img.shape[2]):
            features.append(hog(img[:,:,channel],
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=True,
                                visualise=False, feature_vector=True))

        features = np.ravel(features)
    else:
        features = hog(img[:,:,hog_channel],
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=True,
                                visualise=False, feature_vector=True)

    return features

def extract_features(img,
                    enable_spatial_features = False, spatial_size = (32, 32),
                    enable_histogram_features = False, hist_bins = 32, bins_range = (0, 256),
                    enable_hog_features = True, orient = 9, pix_per_cell=8, cell_per_block=5, hog_channel=0):

    features = []

    if enable_spatial_features == True:
        spatial_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        spatial_features = bin_spatial(spatial_img, spatial_size)
        features.append(spatial_features)

    if enable_histogram_features == True:
        histogram_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        histogram_features = color_histogram(histogram_img, hist_bins, bins_range)
        features.append(histogram_features)

    if enable_hog_features == True:
        hog_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        hog_features = extract_hog_features(hog_img, orient, pix_per_cell, cell_per_block, hog_channel)
        features.append(hog_features)

    features = np.concatenate(features)
    return features

enable_spatial_features = False
spatial_size = (16, 16)
enable_histogram_features = False
hist_bins = 16
bins_range = (0, 256)
enable_hog_features = True
orient = 8
pix_per_cell = 8
cell_per_block = 1
hog_channel = "ALL"

train = False
svc = None
X_scaler = None

if train == True:

    for car in cars:
        img = cv2.imread(car)
        features = extract_features(img, enable_spatial_features, spatial_size, enable_histogram_features, hist_bins, bins_range, enable_hog_features, orient, pix_per_cell, cell_per_block, hog_channel)
        car_features.append(features)

    for not_car in not_cars:
        img = cv2.imread(not_car)
        features = extract_features(img, enable_spatial_features, spatial_size, enable_histogram_features, hist_bins, bins_range, enable_hog_features, orient, pix_per_cell, cell_per_block, hog_channel)
        not_car_features.append(features)

###################################################################################################
## Prepare data for training
###################################################################################################

    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    print(X.shape)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((
            np.ones(len(car_features)),
            np.zeros(len(not_car_features))))

    random_state = np.random.randint(0, 100)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=random_state)

###################################################################################################
##  Train SVM
###################################################################################################

    svc = LinearSVC()

    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t1, 2), ' seconds to train SVC...')

    print('Train accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
    print('Test accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    pickle.dump(svc, open("svc.p", "wb"))
    pickle.dump(X_scaler, open("x_scaler.p", "wb"))

else:

    svc = pickle.load(open("svc.p", "rb"))
    X_scaler = pickle.load(open("x_scaler.p", "rb"))

###################################################################################################
## Sliding windows
###################################################################################################

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def draw_windows(img, windows, color, thickness):
    image = np.copy(img)
    for window in windows:
        cv2.rectangle(image, window[0], window[1], color, thickness)
    return image

###################################################################################################
## Heatmap
###################################################################################################

def add_heat(heatmap, windows):
    for window in windows:
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 4
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] *= 2
    return heatmap

def decay_heat(heatmap, decay):
    heatmap *= decay
    return heatmap

def threshold_heatmap(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_boxes(img, labels):
    for car in range(1, labels[1]+1):
        nonzero = (labels[0] == car).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img

###################################################################################################
## Pipeline for single image
###################################################################################################

scaler = X_scaler
classifier = svc
last_hmap = None

def pipeline(img, video=True):
    

    x_start_stops = [(0, 1280), (0, 1280), (0, 1280), (0, 1280), (0, 1280)]
    y_start_stops = [(400, 656), (400, 656), (400, 656), (400, 656), (400, 656)]
    scales = [(48, 48), (64, 64), (96, 96), (128, 128), (192, 192)]
    overlaps = [(0.25, 0.25), (0.25, 0.25), (0.5, 0.5), (0.85, 0.85), (0.85, 0.85)]

    windows = []
    for scale, overlap, x_start_stop, y_start_stop in zip(scales, overlaps, x_start_stops, y_start_stops):
        scaled_windows = slide_window(img, x_start_stop, y_start_stop, scale, overlap)
        windows.extend(scaled_windows)

    detected_windows = []

    for window in windows:

        window_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        features = extract_features(window_img, enable_spatial_features, spatial_size, enable_histogram_features, hist_bins, bins_range, enable_hog_features, orient, pix_per_cell, cell_per_block, hog_channel)
        features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = classifier.predict(features)

        if prediction == 1:
            detected_windows.append(window)

    windows_img = np.copy(img)
    #windows_img = draw_windows(windows_img, windows, color = (0, 0, 255), thickness = 6)

    detections_img = np.copy(windows_img)
    detections_img = draw_windows(detections_img, detected_windows, color = (0, 255, 0), thickness = 8)

    global last_hmap

    if last_hmap == None:
        hmap = np.zeros_like(img[:,:,0]).astype(np.float)
    else:
        hmap = last_hmap

    hmap = decay_heat(hmap, 0.5)

    hmap = add_heat(hmap, detected_windows)
    hmap = np.clip(hmap, 0, 255)
    last_hmap = hmap

    hmap_threshold = threshold_heatmap(hmap, 32)
    hmap_labels = label(hmap_threshold)
    refined_img = draw_boxes(np.copy(img), hmap_labels)

    hmap_rgb = np.uint8(np.dstack((hmap, hmap, hmap)))
    result = cv2.addWeighted(refined_img, 1.0, hmap_rgb, 0.8, 0.5)

    return result

###################################################################################################
## Process test images
###################################################################################################

test_images = glob.glob('test_images/*.jpg')
test_images = []
for test_image in test_images:
    image = mpimg.imread(test_image)
    result = pipeline(image)
    fig = plt.figure()
    plt.subplot(141)
    plt.imshow(result[0])
    plt.subplot(142)
    plt.imshow(result[1])
    plt.subplot(143)
    plt.imshow(result[2], cmap='hot')
    plt.subplot(144)
    plt.imshow(result[3])
    fig.tight_layout()
    plt.show()

###################################################################################################
## Process video
###################################################################################################

clip_output_filename = 'project_video_detection.mp4'
clip_input = VideoFileClip('project_video.mp4').subclip(10, 20)
clip_output = clip_input.fl_image(pipeline)
clip_output.write_videofile(clip_output_filename, audio=False)
