import csv
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout
from keras.models import Sequential
from scipy import ndimage
import numpy as np

epochs = 7

# CSV reading from "04 - Training The Network" course chapter
def loadTrainingData():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip the header line, if any
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        # csv columns:
        # center,left,right,steering,throttle,brake,speed
        source_path = line[0]
        # TODO use left and right images if required, line[1] & line[2], with adjusted steering
        # TODO mirror images and reverse steering-input to remove left-bias
        # TODO drop some straight-forward data points, overrepresented
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3]) # line[3] is a string
        measurements.append(measurement)
    X_train = np.array(images)
    y_train = np.array(measurements)
    print('Loaded! Nr images: {}, nr measurements: {}'.format(len(images), len(measurements)))
    return X_train, y_train



# Below initially copied from the video in the course materials:
# 14. Even More Powerful Network - 11 - NVIDIA architecture
# Creates the network described at
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/#attachment_7025
# Modified to shrink down the width to be similar to nvidia's original architecture
def model():
    model = Sequential()
    # Normalize and crop the input images
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

    # How many units should be trimmed off at the beginning and end of
    # the 2 cropping dimensions (width, height).
    # This means output_shape=(65, 320, 3)
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Nvidia's sample has an input here of 66x200, however our input is 65x320
    # subsampling/striding (2,3) for the first layer instead of (2,2)
    # and using a wider kernel (5,7 instead of 5,5)
    # originally: model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    #model.add(Convolution2D(24, 5, 7, subsample=(2,3), activation="elu"))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))

    # input 31x158 (nvidia: 31x98)
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu"))

    # input 14x77 (nvidia: 14x47)
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu"))

    # input 5x37 (nvidia: 5x22)
    model.add(Convolution2D(64, 3, 3, activation="elu"))

    # input 3x35 (nvidia: 3x20)
    model.add(Convolution2D(64, 3, 3, activation="elu"))

    # input 64x1x33 (nvidia: 64x1x18)
    model.add(Flatten())
    # nvidia: flattened to 1164 according to figure, 1152 according to math
    # flattens to 2112 with kernel (5,5) & subsample=(2,2) in first layer
    # flattens to 1280 with kernel (5,7) & subsample=(2,3) in first layer
    # flattens to  832 with kernel (5,9) & subsample=(2,4) in first layer

    # after flattening there should be around 1164 neurons (NVidia's original arch),
    model.add(Dense(100, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

# dummy-model from course materials - "Training the Network"
def dummyModel():
    model = Sequential()
    model.add(Flatten(input_shape=[160,320,3]))
    model.add(Dense(1))
    return model


def trainModel(model, X_data, y_data):
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_data, y_data, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    return model

def saveModel(model):
    model.save('model.h5')

def main():
    X_train, y_train = loadTrainingData()
    m = model()
    m = trainModel(m, X_train, y_train)
    saveModel(m)

main()
