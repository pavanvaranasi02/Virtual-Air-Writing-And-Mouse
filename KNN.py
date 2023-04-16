# K-Nearest Neighbors code.
import cv2
import pandas as pd
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('')

# # Preprocess the data to get it ready for training by separating the input features (pixel values) and the target variable (label) into separate arrays.
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

from sklearn.neighbors import KNeighborsClassifier

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# # Train the model on the training data
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

# # Test the model on the testing data
y_pred = knn.predict(X_test)

# # Print the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# # Accuracy of K-Nearest Neighbors is: 0.9862934622096926 (for n_neighbors = 1)
import pickle

# train your KNN model and assign it to the variable "model"

# save the KNN model using pickle
filename = 'knn_model.sav'
pickle.dump(knn, open(filename, 'wb'))

# Load the model
# with open('model.pickle', 'rb') as f:
#     model = pickle.load(f)