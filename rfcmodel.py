import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from directory of character images and labels
data_dir = './chars/'
X = []
y = []
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        img = Image.open(os.path.join(label_dir, filename)).convert('L')
        img = img.resize((100, 100))
        # img.thumbnail((100, 100), Image.ANTIALIAS)
        img_arr = np.array(img).flatten()
        X.append(img_arr)
        y.append(label)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing data
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"Random Forest Classifier accuracy: {score}")