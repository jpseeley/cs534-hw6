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
from keras.layers import Dense, Activation, Dropout 
from keras.constraints import MaxNorm
from keras import initializers
from keras import utils as np_utils
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import copy
import random
import seaborn as sn
import pandas as pd

# Model Template

## Pre-Processing ##
# Read in numpy files
# Have two 6500 x 28 x 28 arrays
images = np.load('images.npy')
labels = np.load('labels.npy')

# Reshape into 6500 x 1 x 784 
flat_images = images.reshape(len(images), 1, len(images[0])*len(images[0]))
onehot_labels = np_utils.to_categorical(labels)

# Could pre-process data to do binary detection on black and white pixels
#  See Kay VII, p. 128 - 131 (178-181)

## Helper functions

# Plots flattened image
def plot_num(image,label):
	plt.imshow(image.reshape(28, 28))
	plt.title('label={}'.format(np.argmax(label)))
	plt.show()


## Split data into training, validation, and test sets ##
# Using stratified sampling to maintain relative frequency
# Set distribution is roughly as follows:
#  ~60% - Training
#  ~15% - Validation
#  ~25% - Testing

# @review
# Procedure 
#  1. Split data into each of the digit classes (0 - 9)
#  2. At random put 60% of data set into training set
#  3. At random put 15% into validation set
#  4. At random put remaining 25% into testing set

copy_flat_images = copy.deepcopy(flat_images).tolist()
copy_labels = copy.deepcopy(labels).tolist()

# print(copy_flat_images[0])
print((np.asarray(copy_flat_images)).shape)

# Sets
training_images = [[]]
training_labels = [-1]
validation_images = [[]]
validation_labels = [-1]
testing_images = [[]]
testing_labels = [-1]

# Stratified Sampling
# Breaking data set into 10 groups for each digit
# For our data set
# 0: 651
# 1: 728
# 2: 636
# 3: 669
# 4: 654
# 5: 568
# 6: 664
# 7: 686
# 8: 600
# 9: 644
grouped_flat_images = [[],[],[],[],[],[],[],[],[],[]] # empty 10 x _ list
for i in range(len(copy_flat_images)):
	curr_label = copy_labels[i] # used as index into grouped images
	curr_image = copy_flat_images[i]

	if not grouped_flat_images[curr_label]:
		grouped_flat_images[curr_label] = [curr_image]
	else:
		grouped_flat_images[curr_label].append(curr_image)

# @debug
# for group in grouped_flat_images:
# 	# plot_num(np.asarray(group[0]).reshape(28, 28)) 
# 	# print(len(group))
# 	i = 0
# 	for img in group:
# 		if isinstance(img[0], int):
# 			print("{} @@@@@@@@@@@@".format(i))
# 			print(img[0])
# 		i+=1

rand_label = 0
for group in grouped_flat_images:
	# Limits for each set
	curr_group = group
	per_60 = int(0.6*len(curr_group))
	per_15 = int(0.15*len(curr_group))
	per_25 = len(curr_group) - per_60 - per_15
	print("{}: {}".format(rand_label, len(group)))

	#  Training Set
	for i in range(per_60): # want to perform this 60% of our dataset
		# Get random image and label from dataset
		rand_index = random.randint(0, len(curr_group)-1)
		# print("{}/{}: {} from [{},...,{}]".format(i, per_60-1, rand_index, 0, len(curr_group)-1))
		rand_image = curr_group[rand_index] # image
		del curr_group[rand_index] # delete image

		# Add the image and label to our training set
		if not training_images[0]: # empty training set
			training_images[0] = rand_image
			# print(rand_image)
		else: # non-empty training set
			training_images.append(rand_image)
		if training_labels[0] == -1: # empty training set
			training_labels[0] = rand_label
		else: # non-empty training set
			training_labels.append(rand_label)

	#  Validation Set
	for i in range(per_15): # want to perform this 15% of our dataset
		# Get random image and label from dataset
		rand_index = random.randint(0, len(curr_group)-1)
		# print("{}/{}: {} from [{},...,{}]".format(i, per_60-1, rand_index, 0, len(curr_group)-1))
		rand_image = curr_group[rand_index] # image
		del curr_group[rand_index] # delete image

		# Add the image and label to our training set
		if not validation_images[0]: # empty training set
			validation_images[0] = rand_image
			# print(rand_image)

		else: # non-empty training set
			validation_images.append(rand_image)
		if validation_labels[0] == -1: # empty training set
			validation_labels[0] = rand_label
		else: # non-empty training set
			validation_labels.append(rand_label)

	#  Testing Set
	for i in range(per_25): # want to perform this 25% of our dataset
		# Get random image and label from dataset
		rand_index = random.randint(0, len(curr_group)-1)
		# print("{}/{}: {} from [{},...,{}]".format(i, per_60-1, rand_index, 0, len(curr_group)-1))
		rand_image = curr_group[rand_index] # image
		del curr_group[rand_index] # delete image

		# Add the image and label to our training set
		if not testing_images[0]: # empty training set
			testing_images[0] = rand_image
			# print(rand_image)

		else: # non-empty training set
			testing_images.append(rand_image)
		if testing_labels[0] == -1: # empty training set
			testing_labels[0] = rand_label
		else: # non-empty training set
			testing_labels.append(rand_label)

	rand_label += 1

# @debug
for i in range(len(training_images)):
	# if i == 0:
		# print(training_images[0])
	if isinstance(training_images[i][0], int):
		print(training_images[i])
	# print("{}: {} x {}".format(i, len(training_images[i]), len(training_images[i][0])))

# Final reshaping for correct format for neural network
training_labels = np.reshape(np_utils.to_categorical(np.asarray(training_labels)), (len(training_labels), 1, 10))
validation_labels = np.reshape(np_utils.to_categorical(np.asarray(validation_labels)), (len(validation_labels), 1, 10))
testing_labels = np.reshape(np_utils.to_categorical(np.asarray(testing_labels)), (len(testing_labels), 1, 10))
training_images = np.asarray(training_images)
validation_images = np.asarray(validation_images)
testing_images = np.asarray(testing_images)

# @debug
# Printing out final array shapes
print(training_images.shape)
print(training_labels.shape)
print(validation_images.shape)
print(validation_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)


## Neural Network Model ##
# Initialize weights randomly for every layer
# Activation Units: ReLu, SeLu, & Tanh
# Experiment with: 
#  Number of layers
#  Number of neurons / layer (including 1st layer)
# Do not change final layer
model = Sequential() # declare model
model.add(Dense(512, input_shape=(1, 28*28), kernel_initializer=initializers.random_normal(stddev=1/512), kernel_constraint=MaxNorm(4.5))) # first layer
model.add(Activation('selu'))
## @todo
model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)))
# model.add(Dropout(0.2))
model.add(Dense(512, activation='selu', kernel_initializer='he_normal', kernel_constraint=MaxNorm(4.0)))
model.add(Dropout(0.15))
model.add(Dense(512, activation='selu', kernel_initializer='he_normal', kernel_constraint=MaxNorm(3.5)))
model.add(Dropout(0.1))

# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(10))
#
##
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# ## Compile and Train Neural Network Model ## 
# # Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# # Train Model
x_train = training_images.astype('float32')/255
y_train = training_labels
x_val = validation_images.astype('float32')/255
y_val = validation_labels

# Can vary epochs + batch_size
# 1. increased epochs to 20 to make sure we reach asymtoptic validation accuracy
# 2. batch size higher actually reduced performance, likely because it was higher than val
#   a. lower however has higher variance through epochs, potentially better accuracy ~300 sweet spot
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=10, 
                    batch_size=300)


## Report Results of Training and Validation ##
# Printout
# print(history.history) @todo (uncomment for now)

# Prediction
x_test = testing_images.astype('float32')/255
# y_test = np.reshape(testing_labels, (len(x_test), 10))
y_test = np.argmax(testing_labels, axis=2)
y_pred = model.predict(x_test) # uses the test set @todo

# Confusion Matrix @todo
# Plots very simple confusion matrix
# y_pred = np.reshape(np_utils.to_categorical(np.argmax(y_pred, axis=2)), (len(y_test), 10)) # turn pred binary and reshape
y_pred = np.argmax(y_pred, axis=2)
print(y_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred) # labels=[str(i) for i in range(10)]
df_cm = pd.DataFrame(cm, range(10), range(10))
plt.figure(figsize=(10,10))
sn.heatmap(df_cm, annot=True)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()

# print(np.argmax(y_pred, axis=2))


# @debug
# for i in range(10):
# 	base = 900
# 	plot_num(x_train[i+base],y_train[i+base])
# 	plot_num(x_val[i+base],y_val[i+base])







