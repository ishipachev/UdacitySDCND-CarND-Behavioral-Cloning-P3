import csv
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from sklearn.model_selection import train_test_split
import sklearn


def read_folder(folder_path):
    csv_file_path = os.path.join(folder_path, 'driving_log.csv')
    img_folder = os.path.join(folder_path, 'IMG')
    lines = []
    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = os.path.join(img_folder, filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    return (images, measurements)

def read_samples(folder_path):
    csv_file_path = os.path.join(folder_path, 'driving_log.csv')
    lines = []
    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


# folder_path = 'data_collected/try1'
# (images1, measurements1) = read_folder(folder_path)
#
# folder_path = 'data_collected/try2'
# (images2, measurements2) = read_folder(folder_path)
#
# folder_path = 'data_collected/try3_restore'
# (images3, measurements3) = read_folder(folder_path)
#
# images = images1 + images2 + images3
# measurements = measurements1 + measurements2 + measurements3
#
# X_train = np.array(images)
# y_train = np.array(measurements)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size//2):
            batch_samples = samples[offset:offset+batch_size//2]
            images = []
            angles = []
            for batch_sample in batch_samples:
                img_path = batch_sample[0]
                # filename = source_path.split('/')[-1]
                # current_path = os.path.join(img_folder, filename)
                # # name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(img_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # aug_image = center_image.copy()
                # aug_image = cv2.flip(aug_image, 1)
                # aug_angle = -center_angle
                images.append(cv2.flip(center_image, 1))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # yield sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)

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
    # model.add(Dropout(0.3))
    # model.add(Dense(50))
    # model.add(Dropout(0.3))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dense(1))
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(X_train, y_train, validation_split=0.2,
    #           shuffle=True, nb_epoch=20, verbose=2)
    return model


samples = read_samples('data_collected/try1') + \
          read_samples('data_collected/try2') + \
          read_samples('data_collected/try3_restore') + \
          read_samples('data_collected/try4') + \
	  read_samples('data_collected/try5')
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# model = simple_network()
model = nvidia_network()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3)

model.save('output/aug_model.h5')

