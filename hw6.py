# Homework 6

# We are using MNIST dataset
# Each image is 28x28 greyscale (0-255 color range)
#	0: white
# 	255: black
# images.npy holds the images
# labels.npy holds the labels (0-9)

# Pre-processing?
# Using Numpy

# Artificial Neural Network
# Using Keras

# Decision Tree
# Scikit-learn package

# Our Model
# Input: Single image 28x28
# Output: Number 0-9
# Model should be saved as "trained_model.hw6"

### START OF MODEL TEMPLATE GIVEN IN ASSIGNMENT ###
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# Model Template

# Read in numpy files

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
#
#
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=10, 
                    batch_size=512)


# Report Results

print(history.history)
model.predict()