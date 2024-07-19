# Potato Disease Identifier

This project involves a convolutional neural network (CNN) model developed using TensorFlow and Keras to identify potato diseases from images. The model classifies potato images into different disease categories to aid in agricultural management.

## Requirements

- Python 3.10.14
- TensorFlow 2.10
- NumPy
- Matplotlib

## Dataset

The dataset used in this project is from the PlantVillage collection and includes images of potato plants categorized into the following classes:
- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

Ensure that the dataset is organized into separate directories for each class.

## Model Architecture

The CNN model used in this project includes the following layers:
1. **Conv2D**: 32 filters, kernel size (3, 3), ReLU activation
2. **MaxPooling2D**: Pool size (2, 2)
3. **Conv2D**: 64 filters, kernel size (3, 3), ReLU activation
4. **MaxPooling2D**: Pool size (2, 2)
5. **Conv2D**: 64 filters, kernel size (3, 3), ReLU activation
6. **MaxPooling2D**: Pool size (2, 2)
7. **Conv2D**: 64 filters, kernel size (3, 3), ReLU activation
8. **MaxPooling2D**: Pool size (2, 2)
9. **Conv2D**: 64 filters, kernel size (3, 3), ReLU activation
10. **MaxPooling2D**: Pool size (2, 2)
11. **Conv2D**: 64 filters, kernel size (3, 3), ReLU activation
12. **MaxPooling2D**: Pool size (2, 2)
13. **Flatten**
14. **Dense**: 64 units, ReLU activation
15. **Dense**: 3 units (one for each class), Softmax activation

## Training

- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: SparseCategoricalCrossentropy
- **Metrics**: Accuracy

The dataset is split into:
- **Training Set**: 80%
- **Validation Set**: 10%
- **Test Set**: 10%

Data augmentation techniques such as random flips and rotations are applied to the training set.

## Usage

1. **Prepare the Data**: Place the dataset in a directory named `PlantVillage`, with subdirectories for each class.
2. **Run the Script**: Execute the Python script provided to train and evaluate the model.
3. **Evaluate**: The model's performance will be displayed upon evaluation on the test set.

## Code

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Parameters
IMAGE_SIZE = 256
BATCH_SIZE = 32 
CHANNELS = 3
EPOCHS = 50

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

# Data augmentation and preprocessing
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])
train_ds = dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define model
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    tf.keras.Input(shape=input_shape), 
    resize_and_rescale,
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    verbose=1,
    validation_data = val_ds
)

# Evaluate the model
scores = model.evaluate(test_ds)
print("Test Loss:", scores[0])
print("Test Accuracy:", scores[1])
