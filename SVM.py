import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# y_train = y_train.ravel()

# Define the SVM model
model = svm.SVC(kernel='linear', C=1)

# Train the model on the training data
model.fit(X_train, y_train)

# Use the model to predict the labels for the test data
y_pred = model.predict(X_test)

# Compute the accuracy of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot some examples of air writing and their predicted labels
n_examples = 5
for i in range(n_examples):
    # Choose a random example from the test set
    X_test = np.array(X_test)
    idx = np.random.randint(X_test.shape[0])
    x = X_test[idx]
    y_true = y_test[idx]

    # Predict the label for the example
    y_pred = model.predict(x.reshape(1, -1))

    # Reshape the input vector as an image and plot it
    img = x.reshape((100, 100))
    plt.subplot(n_examples, 1, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"True label: {y_true}, Predicted label: {y_pred}")

plt.show()