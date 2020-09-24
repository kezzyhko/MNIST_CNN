# imports
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GaussianNoise, BatchNormalization
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import zipfile
from sklearn.metrics import f1_score
import os
import matplotlib.pyplot as plt



# load the dataset

x = np.load('x.npy')
y = np.load('y.npy')

print(x.shape)
print(y.shape)



# data augmentation

# plt.imshow(x[0].reshape(28, 28)/255., cmap='Greys')
g = GaussianNoise(40)
x = g(x.astype('float32'), training=True)
plt.imshow(tf.reshape(x[0], (28, 28))/255., cmap='Greys')

generator = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.3,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    validation_split = 0.15
)



# create the model

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides = 2, padding='same'))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides = 2, padding='same'))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides = 2, padding='same'))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=[Recall(), Precision()])



# train the model

# model.fit(x, pd.get_dummies(y).astype(np.float32), validation_split=0.15)

model.fit(
    generator.flow(x, pd.get_dummies(y).astype(np.float32), batch_size = 128),
    validation_data = generator.flow(x, pd.get_dummies(y).astype(np.float32), subset = 'validation'),
    steps_per_epoch = len(x) / 128,
    epochs = 10
)



# save the model

model.save('model.h5')
zipfile.ZipFile('model.h5.zip', mode='w').write("model.h5")



# make the prediction
y_pred = model.predict(x)

# transform prediction to labels
y_pred_labels = np.argmax(y_pred, axis=1).astype(np.uint8)

# evaluate performance of the model
tf.print(f1_score(y_pred_labels, y, average='micro'))