'''
The algorithm used in this code is ResNet50, which is a deep convolutional neural network 
architecture used for image classification. It is pre-trained on the ImageNet dataset and can 
be fine-tuned for specific image classification tasks. The code also uses the Adam optimizer and 
categorical cross-entropy loss function for training the model.
'''

from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Input
from keras.models import Model
from keras.applications import resnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)

datagen = ImageDataGenerator(width_shift_range=[-10, 10], height_shift_range=[-10, 10],
                             rotation_range=30, zoom_range=[.8, 1.2], rescale=1/255.)
gen = datagen.flow_from_directory(directory='./chars/', target_size=(100, 100), color_mode='grayscale',
                                  batch_size=8, interpolation='nearest')


a_file = open("model_loader.pkl", "rb")
dict_file = pickle.load(a_file)
a_file.close()
gen.class_indices = dict_file

model = resnet.ResNet50(input_shape=(100, 100, 1), weights=None, classes=gen.num_classes)
opt = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True)

hist = model.fit_generator(gen, epochs = 30)
model.save('model.h5', include_optimizer=False)

loss = hist.history['loss']
acc = hist.history['accuracy'] # error
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='Loss')
plt.plot(epochs, acc, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.legend()
plt.show()
