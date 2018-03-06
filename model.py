import csv
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from sklearn.model_selection import train_test_split
import sklearn


# Read all samples strings from .csv file in list of strings
def read_samples(folder_path):
    csv_file_path = os.path.join(folder_path, 'driving_log.csv')
    lines = []
    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def generator(samples, batch_size=64):
    num_samples = len(samples)
    original_size = batch_size // 2  #for each picture we will also get aumentented one
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, original_size):
            batch_samples = samples[offset:offset+original_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                img_path = batch_sample[0]
                center_image = cv2.imread(img_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                #add augmented image and angle
                images.append(cv2.flip(center_image, 1))
                angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield (X_train, y_train)

# Network for testing purpose
def simple_network():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def nvidia_network():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#Reading collected training data from folders with pictures
samples = read_samples('data_collected/try1') + \
          read_samples('data_collected/try2') + \
          read_samples('data_collected/try3_restore') + \
          read_samples('data_collected/try4') + \
          read_samples('data_collected/try5')

train_samples, validation_samples = train_test_split(samples, test_size=0.1)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

model = nvidia_network()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*2,
                    nb_epoch=3)

model.save('model_fd.h5')

