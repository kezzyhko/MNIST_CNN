# imports
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GaussianNoise, BatchNormalization
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import zipfile
from sklearn.metrics import f1_score
import os
import matplotlib.pyplot as plt



# load the dataset

(mnist_X_train, mnist_y_train), (mnist_X_test, mnist_y_test) = tf.keras.datasets.mnist.load_data()
mnist_X_train = mnist_X_train.reshape((60000, 28, 28, 1))
mnist_X_test = mnist_X_test.reshape((10000, 28, 28, 1))

x = np.concatenate((mnist_X_train, mnist_X_test), axis=0)
y = np.concatenate((mnist_y_train, mnist_y_test), axis=0)

print(x.shape)
print(y.shape)



# data augmentation

generator = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.1,
    zoom_range = (0.9, 2),
    shear_range = 10,
    fill_mode = 'nearest',
    validation_split = 0.15
)



# create the model

model = Sequential()
model.add(GaussianNoise(70, input_shape=(28, 28, 1)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides = 2))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides = 2))

model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=[Recall(), Precision()])



# train the model

# model.fit(x, to_categorical(y), validation_split=0.15)

model.fit(
    generator.flow(x, to_categorical(y), batch_size = 128),
    validation_data = generator.flow(x, to_categorical(y), batch_size = 128, subset = 'validation'),
    steps_per_epoch = len(x) / 128,
    epochs = 15
)



# save the model
model.save('model.h5')
zipfile.ZipFile('model.h5.zip', mode='w').write("model.h5")
print(os.path.getsize("/content/model.h5.zip") / (1024*1024))
files.download('model.h5.zip') 



# make the prediction
y_pred = model.predict(x)

# transform prediction to labels
y_pred_labels = np.argmax(y_pred, axis=1).astype(np.uint8)

# evaluate performance of the model
tf.print(f1_score(y_pred_labels, y, average='micro'))