import cv2
import glob
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

from skimage.feature import hog

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

###################################################################################################
## Load data for cars and not cars
###################################################################################################

cars_dirs = [
        'data/vehicles/GTI_Far',
        'data/vehicles/GTI_Left',
        'data/vehicles/GTI_MiddleClose',
        'data/vehicles/GTI_Right',
        'data/vehicles/KITTI_Extracted']
not_cars_dirs = [
        'data/non-vehicles/Extras',
        'data/non-vehicles/GTI']

cars = []
not_cars = []

for car_dir in cars_dirs:
    print('Loading directory ' + car_dir)
    images = glob.glob(car_dir + '/*.png')
    cars.extend(images)

for not_car_dir in not_cars_dirs:
    print('Loading directory ' + not_car_dir)
    images = glob.glob(not_car_dir + '/*.png')
    not_cars.extend(images)

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

    if hog_channel == 'ALL':

        features = []

        for channel in range(img.shape[2]):
            features.append(hog(img[:,:,channel],
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=True,
                                visualise=False, feature_vector=True))

        features = np.ravel(hog_features)
    else:
        features = hog(img[:,:,hog_channel],
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=True,
                                visualise=False, feature_vector=True)

    return features

def extract_features(img, color_space = 'RGB',
                    spatial_size = (32, 32),
                    hist_bins = 32, bins_range = (0, 256),
                    orient = 9, pix_per_cell=8, cell_per_block=2, hog_channel=0):

    if color_space == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #spatial_features = bin_spatial(img, spatial_size)
    histogram_features = color_histogram(img, hist_bins, bins_range)
    hog_features = extract_hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel)
    features = np.concatenate((histogram_features, hog_features))
    return features

color_space = 'YUV'
spatial_size = (32, 32)
hist_bins = 32
bins_range = (0, 256)
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0

for car in cars:
    img = mpimg.imread(car)
    features = extract_features(img, color_space, spatial_size, hist_bins, bins_range, orient, pix_per_cell, cell_per_block, hog_channel)
    car_features.append(features)

for not_car in not_cars:
    img = mpimg.imread(not_car)
    features = extract_features(img, color_space, spatial_size, hist_bins, bins_range, orient, pix_per_cell, cell_per_block, hog_channel)
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
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
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
## Pipeline for single image
###################################################################################################

def pipeline(img, scaler, classifier):
    
    x_start_stop = [None, None]
    y_start_stop = [400, 656]
    scales = [(48, 48), (64, 64), (96, 96), (128, 128), (192, 192)]
    overlaps = [(0.25, 0.25), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (0.75, 0.75)]

    windows = []
    for scale, overlap in zip(scales, overlaps):
        scaled_windows = slide_window(img, x_start_stop, y_start_stop, scale, overlap)
        windows.extend(scaled_windows)

    detected_windows = []

    for window in windows:

        window_img = test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        features = extract_features(window_img, color_space, spatial_size, hist_bins, bins_range, orient, pix_per_cell, cell_per_block, hog_channel)
        features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = classifier.predict(features)

        if prediction == 1:
            detected_windows.append(window)

    windows_img = np.copy(img)
    windows_img = draw_windows(windows_img, detected_windows, color = (0, 0, 255), thickness = 8)

    return windows_img

###################################################################################################
## Process test images
###################################################################################################

test_images = glob.glob('test_images/*.jpg')
for test_image in test_images:
    image = mpimg.imread(test_image)
    result = pipeline(image, X_scaler, svc)
    plt.imshow(result)
    plt.show()
