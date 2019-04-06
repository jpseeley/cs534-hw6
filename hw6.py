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
from keras import utils as np_utils
from matplotlib import pyplot as plt
import numpy as np
import copy
import random

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

# Plots unflattened image
def plot_num(image):
	plt.imshow(image)
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
		grouped_flat_images[curr_label] = copy.deepcopy([curr_image])
	else:
		grouped_flat_images[curr_label].append(copy.deepcopy(curr_image))

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


# timg = np.asarray(copy_flat_images[0]).reshape(1, 28, 28) @debug

# Deleting and maybe shallow copy issues?, if weird bugs later on, deepcopy everything
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
			training_images[0] = copy.deepcopy(rand_image)
			# print(rand_image)
		else: # non-empty training set
			training_images.append(copy.deepcopy(rand_image))
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
		# if i == 0:
			# print(np.asarray(rand_image).shape)
			# print(rand_image)
		# Add the image and label to our training set
		if not validation_images[0]: # empty training set
			validation_images[0] = copy.deepcopy(rand_image)
			# print(rand_image)

		else: # non-empty training set
			validation_images.append(copy.deepcopy(rand_image))
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
			testing_images[0] = copy.deepcopy(rand_image)
			# print(rand_image)

		else: # non-empty training set
			testing_images.append(copy.deepcopy(rand_image))
		if testing_labels[0] == -1: # empty training set
			testing_labels[0] = rand_label
		else: # non-empty training set
			testing_labels.append(rand_label)

	rand_label += 1

print(len(training_images))
print(len(training_images[0]))
print(len(training_images[0][0]))

# for i in range(len(training_images)):
# # for i in range(2):
# 	img = training_images[i]
# 	box_list = [[]]
# 	# print(img)
# 	for j in range(28):
# 		# print(j)
# 		if not box_list[0]:
# 			box_list[0] = img[0][28*j:28*j+28-1]
# 		else:
# 			box_list.append(img[0][28*j:28*j+28-1])
# 	training_images[i] = box_list
	# print(box_list)
# print(training_images[0])
# print(training_images[1])

for i in range(len(training_images)):
	# if i == 0:
		# print(training_images[0])
	if isinstance(training_images[i][0], int):
		print(training_images[i])
	print("{}: {} x {}".format(i, len(training_images[i]), len(training_images[i][0])))

# training_images = np.asarray(training_images)

# for i in range(len(training_images)):
# 	training_images[i] = copy.deepcopy(np.asarray(training_images[i]))

# print(type(training_images))
# print(type(training_images[0]))
# print(type(training_images[0][0]))

print((np.asarray(training_images)).shape)
print((np.asarray(training_labels)).shape)
print((np.asarray(validation_images)).shape)
print((np.asarray(validation_labels)).shape)
print((np.asarray(testing_images)).shape)
print((np.asarray(testing_labels)).shape)

# print((np.asarray(copy_flat_images)).shape)
# print((np.asarray(copy_labels)).shape)
# print(training_labels)
# training_labels.append(0)
# print(training_labels)



## Neural Network Model ##
# Initialize weights randomly for every layer
# Activation Units: ReLu, SeLu, & Tanh
# Experiment with: 
#  Number of layers
#  Number of neurons / layer (including 1st layer)
# Do not change final layer
model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
# #
# #
# #
# # Fill in Model Here @todo
# #
# #
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# ## Compile and Train Neural Network Model ## 
# # Compile Model
# model.compile(optimizer='sgd',
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])

# # Train Model
# x_train = training_images
# y_train = np_utils.to_categorical(np.asarray(training_labels))
# x_val = validation_images
# y_val = np_utils.to_categorical(np.asarray(validation_labels))

# history = model.fit(x_train, y_train, 
#                     validation_data = (x_val, y_val), 
#                     epochs=10, 
#                     batch_size=512)


# ## Report Results ##
# # Printout
# print(history.history)
# model.predict()

# Confusion Matrix @todo













