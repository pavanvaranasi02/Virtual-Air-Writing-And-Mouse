import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, Lambda
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

train = pd.read_csv("D:\\MiniProject\\DataSet from Eminist\\emnist-letters-train.csv", delimiter=',')
test = pd.read_csv("D:\\MiniProject\\DataSet from Eminist\\emnist-letters-test.csv", delimiter=',')
mapp = pd.read_csv("D:\\MiniProject\\dictfile.txt", delimiter=' ')
print("Train: %s, Test: %s, Map: %s" % (train.shape, test.shape, mapp.shape))

# Constants
HEIGHT = 28
WIDTH = 28

# Split x and y
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
del train

test_x = test.iloc[:, 1:]
test_y = test.iloc[:, 0]
del test

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# Rotate image
def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)
print("train_x:", train_x.shape)

test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)
print("test_x:", test_x.shape)

# Normalize
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

# Number of classes
num_classes = 26

# One hot encoding
train_y = np_utils.to_categorical(train_y - 1, num_classes)
test_y = np_utils.to_categorical(test_y - 1, num_classes)
print("train_y: ", train_y.shape)
print("test_y: ", test_y.shape)

# Reshape image for CNN
train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)
test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)

# Partition into train and val
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.20, random_state=42)

# Building model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(HEIGHT, WIDTH, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.add(Reshape((num_classes,)))
model.add(Lambda(lambda x: x / 0.01))

model.summary()

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

history = model.fit(
    train_x, train_y,
    batch_size=256,
    epochs=30,
    validation_data=(val_x, val_y),
    callbacks=[early_stopping])

score = model.evaluate(test_x, test_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_y, axis=1)

# Print the accuracy of the model
print("Accuracy:", metrics.accuracy_score(y_true_classes, y_pred_classes))
cm = metrics.confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# Save the model
model.save('emnist_mrf_cnn_model.h5')

# Test loss: 0.2035479098558426
# Test accuracy: 0.9311439990997314
# 463/463 [==============================] - 16s 35ms/step
# Accuracy: 0.9311439962159606
# [[736   0   3  13   0   1   4   6   0   3   0   1   1   4   5   1  12   0
#     0   0   6   0   0   0   0   3]
#  [  0 788   0   0   2   0   0   1   1   0   2   1   0   2   0   0   0   1
#     0   0   0   0   0   0   0   2]
#  [  0   0 764   0  16   0   1   0   0   0   0   9   0   0   1   0   0   5
#     2   0   1   1   0   0   0   0]
#  [  1   0   0 773   0   0   0   0   0   2   0   0   0   1  16   5   1   0
#     0   1   0   0   0   0   0   0]
#  [  1   0   4   0 786   1   0   0   1   1   0   1   0   0   0   2   1   1
#     0   1   0   0   0   0   0   0]
#  [  0   0   0   0   3 768   1   0   1   1   0   0   0   0   0   4   0   3
#     2  17   0   0   0   0   0   0]
#  [  8   8   6   1   1   1 681   0   0   5   1   0   0   1   0   0  75   0
#     7   0   0   0   0   0   5   0]
#  [  1   6   0   1   0   0   0 757   0   0   3  12   4  10   0   0   0   0
#     0   1   3   0   0   2   0   0]
#  [  0   0   1   1   1   0   0   0 620  14   0 160   0   0   0   0   0   1
#     0   0   0   2   0   0   0   0]
#  [  0   0   0   1   0   1   0   0  13 776   0   2   0   0   0   0   0   0
#     1   3   1   1   0   0   1   0]
#  [  0   1   0   0   1   0   0   6   0   0 781   2   0   0   0   0   0   1
#     0   4   0   0   0   4   0   0]
#  [  0   0   3   0   1   0   0   2 198   1   0 593   0   0   0   0   0   1
#     0   0   0   0   0   0   1   0]
#  [  0   0   0   0   0   0   0   0   0   0   1   0 795   4   0   0   0   0
#     0   0   0   0   0   0   0   0]
#  [  1   0   0   0   0   0   0  11   0   3   0   1  10 751   0   2   0   8
#     0   0   1   2   7   1   2   0]
#  [  1   1   2  11   0   0   0   0   0   0   0   0   0   1 778   0   0   0
#     0   0   5   1   0   0   0   0]
#  [  0   1   0   4   1   0   0   0   0   0   0   1   0   0   1 792   0   0
#     0   0   0   0   0   0   0   0]
#  [ 11   1   0   1   1   3  91   0   1   1   0   0   0   0   2   0 683   0
#     0   0   1   0   0   0   3   1]
#  [  2   1   2   0   1   0   0   0   4   1   5   1   0   1   0   2   0 766
#     0   3   0   4   0   0   6   1]
#  [  1   1   0   0   0   0   1   0   0   5   0   0   0   0   0   0   0   0
#   392   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0]]