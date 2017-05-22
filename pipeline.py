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

def extract_features(img, spatial_size = (32, 32), hist_bins = 32, bins_range = (0, 256)):
    image = mpimg.imread(img)
    spatial_features = bin_spatial(image, spatial_size)
    histogram_features = color_histogram(image, hist_bins, bins_range)
    features = np.concatenate((spatial_features, histogram_features))
    return features

spatial_size = (32, 32)
hist_bins = 32
bins_range = (0, 256)

for car in cars:
    features = extract_features(car, spatial_size, hist_bins, bins_range)
    car_features.append(features)

for not_car in not_cars:
    features = extract_features(not_car, spatial_size, hist_bins, bins_range)
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
