import csv
import cv2
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense
from keras.models import Sequential
from scipy import ndimage
import numpy as np

epochs = 5

# CSV reading from "04 - Training The Network" course chapter
def loadTrainingData():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip the header line
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        # csv columns:
        # center,left,right,steering,throttle,brake,speed
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train



# Below initially copied from the video in the course materials:
# 14. Even More Powerful Network - 11 - NVIDIA architecture
# Creates the network described at
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/#attachment_7025
def model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # Nvidia's sample has an input here of 66x200, subsample (3,3) or (4,4) first layer instead of (2,2)???
    # and use kernel 2x as big than 5x5?
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    # after flattening there should be around 1164 neurons (NVidia's original arch)
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    return model

def trainModel(model, X_data, y_data):
    model.compile(loss='mse', optimizer='adam')
    # TODO switch to dummy-model and/or allow training
    # ValueError: Error when checking model target:
    #   expected dense_3 to have shape (None, 10) but got array with shape (8036, 1)
    model.fit(X_data, y_data, validation_split=0.2, shuffle=True)
    return model

def saveModel(model):
    model.save('model.h5')

def main():
    X_train, y_train = loadTrainingData()
    m = model()
    m = trainModel(m, X_train, y_train)
    saveModel(m)

main()

