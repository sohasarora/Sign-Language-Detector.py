# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Setup for TensorFlow
print("TensorFlow version:", tf.__version__)

# Prepare data generators for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images to be between 0 and 1
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directory for training and testing
train_data = train_datagen.flow_from_directory('data/train',  # Path to training data
                                              target_size=(64, 64),  # Resize images to 64x64
                                              batch_size=32,  # Number of images to process at once
                                              class_mode='categorical')  # For multi-class classification

test_data = test_datagen.flow_from_directory('data/test',  # Path to testing data
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

# Build the CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # 32 filters of size 3x3
model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling layer to reduce spatial dimensions

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D outputs to 1D
model.add(Flatten())

# Fully connected layers (Dense)
model.add(Dense(128, activation='relu'))  # First fully connected layer
model.add(Dropout(0.5))  # Dropout for regularization (prevents overfitting)

# Output layer (for multi-class classification)
model.add(Dense(train_data.num_classes, activation='softmax'))  # Softmax activation to predict class probabilities

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summarize the model architecture
model.summary()

# Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data)
print(f'Test accuracy: {accuracy*100:.2f}%')

# Save the trained model
model.save('sign_language_model.h5')

# Optionally, plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Load the trained model and make predictions
from tensorflow.keras.preprocessing import image

# Example: Predict a new image (change the path to your image)
img = image.load_img('new_image.jpg', target_size=(64, 64))  # Replace 'new_image.jpg' with your test image
img_array = image.img_to_array(img) / 255.0  # Normalize image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Print the predicted sign language gesture (letter)
print(f'Predicted Sign Language Gesture: {train_data.class_indices[predicted_class]}')

