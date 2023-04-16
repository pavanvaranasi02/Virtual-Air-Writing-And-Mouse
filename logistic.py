# Logistic Regression code.
import cv2
import pandas as pd

data = pd.read_csv('DataSet from Kaggle\A_Z Handwritten Data\A_Z Handwritten Data.csv')

# # Preprocess the data to get it ready for training by separating the input features (pixel values) and the target variable (label) into separate arrays.
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# # Split the data into training and testing sets using train_test_split from sklearn.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Normalize the input data to be between 0 and 1, since the pixel values can range from 0 to 255.

X_train = X_train / 255.0
X_test = X_test / 255.0

# Logistic Regression model on the A-Z Handwritten Data:

from sklearn.linear_model import LogisticRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train the model on the training data
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Test the model on the testing data
y_pred = classifier.predict(X_test)

# Print the accuracy of the model
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Accuracy of logistic regression is: 0.8801584105249026