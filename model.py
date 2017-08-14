from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense
from keras.models import Sequential

epochs = 5


# Below initially copied from the video in the course materials:
# 14. Even More Powerful Network - 11 - NVIDIA architecture
# Creates the network described at
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/#attachment_7025
def model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # Nvidia's sample has an input here of 66x200, subsample (3,3) or (4,4) first layer instead of (2,2)???
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    return model

