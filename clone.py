import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
csv_path = ('/media/nort/StorageHDD/work/sdc/project2/CarND-Behavioral-Cloning-P3/data_collected/try1/driving_log.csv')
img_folder = ('/media/nort/StorageHDD/work/sdc/project2/CarND-Behavioral-Cloning-P3/data_collected/try1/IMG/')

with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = img_folder + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('output/try1_model.h5')

