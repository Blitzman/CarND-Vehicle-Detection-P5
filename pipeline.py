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

## TODO

###################################################################################################
## Prepare data for training
###################################################################################################

X = np.vstack((car_features, not_car_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform()

y = np.hstack((
        np.ones(len(car_features)),
        np.zeros(len(not_car_features))))

random_state = np.random.randint(0, 100)
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size, random_state)

###################################################################################################
## Train SVM
###################################################################################################

## TODO
