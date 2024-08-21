import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Define the categories
Categories = ['cats', 'dogs']

# Initialize the input and output arrays
flat_data_arr = []
target_arr = []

# Define the path to the dataset
datadir = 'IMAGES/'

# Load and preprocess the images
for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

# Convert the input and output arrays to numpy arrays
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.2, random_state=42)

# Define the SVM model
model = svm.SVC()

# Define the hyperparameter grid for tuning
param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [1, 10, 100]}

# Perform grid search to find the optimal hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Print the best hyperparameters and the corresponding accuracy
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Accuracy: {grid_search.best_score_}')

# Train the SVM model with the optimal hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')