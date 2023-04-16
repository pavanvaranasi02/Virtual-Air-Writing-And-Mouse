import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('DataSet from Kaggle\\A_Z Handwritten Data\\A_Z Handwritten Data.csv')

# Preprocess the data to get it ready for training by separating the input features (pixel values) and the target variable (label) into separate arrays.
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train an RFC model with the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Test the model on the testing data
y_pred = rfc.predict(X_test)

# Print the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model to a file
filename = 'rfc_model.sav'
joblib.dump(rfc, filename)


# Accuracy: 0.9873942811115586