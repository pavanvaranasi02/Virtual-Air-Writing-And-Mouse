import cv2
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('D:\\MiniProject\\DataSet from Kaggle\\A_Z Handwritten Data\\A_Z Handwritten Data_updated.csv') 

# Preprocess the data to get it ready for training by separating the input features (pixel values) and the target variable (label) into separate arrays.
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Reshape the input features to 28x28 grayscale images
X = X.reshape(-1, 28, 28, 1)

# One-hot encode the target variable
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Define a CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(28, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Test the model on the testing data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print the accuracy of the model
print("Accuracy:", accuracy_score(y_true_classes, y_pred_classes))

# Save the model to a file
model.save('cnn_model.h5')


# Accuracy: 0.984871874035196
