
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Reshape the dataset to have a single color channel
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize the pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert the labels to categorical (one-hot encoding)
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
print(model.summary())
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=50, validation_split=0.2, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=1, batch_size=64)


# Save the model
model.save('mnist.keras')