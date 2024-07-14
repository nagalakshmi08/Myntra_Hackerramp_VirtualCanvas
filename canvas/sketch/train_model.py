# matchapp/train_model.py

import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
import os

# Load the dataset
dataset_path = os.path.join(os.path.dirname(__file__), '../../archive/samp_sub.csv')
data = pd.read_csv(dataset_path)

# Assuming your CSV has a column 'image_name' that contains paths to images
image_paths = data['image_name'].values
labels = data['category'].values  # Assuming there is a label column

# Load and preprocess images
def load_and_preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        img_path_full = os.path.join(os.path.dirname(__file__), '../../archive/', img_path)
        img = cv2.imread(img_path_full)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)
    return np.array(images)

images = load_and_preprocess_images(image_paths)

# Split the data
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
train_labels = pd.get_dummies(train_labels).values
val_labels = pd.get_dummies(val_labels).values

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_labels.shape[1], activation='softmax')(x)  # Ensure the correct number of classes

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10)

# Save the model
model.save(os.path.join(os.path.dirname(__file__), 'sketch_model.h5'))

# Function to extract features using the trained model
def extract_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features

# Extract features from dataset images
dataset_features = np.array([extract_features(img, model) for img in images])

# Save the features and image paths
np.save(os.path.join(os.path.dirname(__file__), 'dataset_features.npy'), dataset_features)
np.save(os.path.join(os.path.dirname(__file__), 'image_paths.npy'), image_paths)
