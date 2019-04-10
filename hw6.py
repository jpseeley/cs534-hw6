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
from sklearn.tree import DecisionTreeClassifier
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

print(images.shape)
print(labels.shape)

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

def plot_confusion_matrix(cm_):
	plt.figure(figsize=(10,10))
	ax = plt.gca()
	df_cm = pd.DataFrame(cm_, range(10), range(10))
	sn.heatmap(df_cm, annot=True, fmt='d')
	plt.xlabel('True Label')
	plt.ylabel('Predicted Label')
	plt.title('Confusion Matrix: Acc={:0.3f}%'.format(100*np.trace(cm_)/np.sum(cm_)))
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
print((np.asarray(copy_labels)).shape)

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
## 
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


## Compile and Train Neural Network Model ## 
# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
x_train = training_images.astype('float32')/255
y_train = training_labels
x_val = validation_images.astype('float32')/255
y_val = validation_labels

# Shuffle training set randomly because it is ascending right now
train_shuffler = np.arange(0, len(x_train), 1)
np.random.shuffle(train_shuffler)
x_train = x_train[train_shuffler]
y_train = y_train[train_shuffler]
# Shuffle validation set randomly because it is ascending right now
val_shuffler = np.arange(0, len(x_val), 1)
np.random.shuffle(val_shuffler)
x_val = x_val[val_shuffler]
y_val = y_val[val_shuffler]

# Can vary epochs + batch_size
# 1. increased epochs to 20 to make sure we reach asymtoptic validation accuracy
# 2. batch size higher actually reduced performance, likely because it was higher than val
#   a. lower however has higher variance through epochs, potentially better accuracy ~300 sweet spot
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=1, 
                    batch_size=300)

# @todo - write model to file

## Report Results of Training and Validation ##
# Printout
print(history.history)

# Prediction
x_test = testing_images.astype('float32')/255
# y_test = np.argmax(testing_labels, axis=2)
y_test = testing_labels

# Shuffling testing set
test_shuffler = np.arange(0, len(x_test), 1)
np.random.shuffle(test_shuffler)
x_test = x_test[test_shuffler]
y_test = y_test[test_shuffler]

y_pred = model.predict(x_test)

print(y_test.shape)
print(y_pred.shape)

# Confusion Matrix 
# Plots simple confusion matrix using seaborn/pandas + matplotlib
# y_pred = np.argmax(y_pred, axis=2)
ann_cm = confusion_matrix(np.argmax(y_test, axis=2), np.argmax(y_pred, axis=2)) 
# plot_confusion_matrix(ann_cm)

# print(cm)

## Decision Tree ##
# Reshape all data to 2 dimensions
# x_ => YYY x 748    
# y_ => YYY x 1      - no longer one-hot
x_train = np.reshape(x_train, (len(x_train), len(x_train[0][0])))
y_train = np.argmax(np.reshape(y_train, (len(y_train), len(y_train[0][0]))), axis=1)
x_val = np.reshape(x_val, (len(x_val), len(x_val[0][0])))
y_val = np.argmax(np.reshape(y_val, (len(y_val), len(y_val[0][0]))), axis=1)
x_test = np.reshape(x_test, (len(x_test), len(x_test[0][0])))
y_test = np.argmax(np.reshape(y_test, (len(y_test), len(y_test[0][0]))), axis=1)

# Baseline Decision Tree
# Accuracy against validation set:
#  1. 78.703%
#  2. 76.438% 
#  3. 77.295%
# Around 76-78%, lots of room for improvement
# Default have main params of:
# min_samples_split=2, min_samples_leaf=1, max_depth=None, max_leaf_nodes=None
baseline_classifier = DecisionTreeClassifier()
baseline_classifier_fit = baseline_classifier.fit(x_train, y_train)
baseline_tree_prediction = baseline_classifier_fit.predict(x_val)
baseline_tree_confusion_matrix = confusion_matrix(y_val, baseline_tree_prediction)
# print(baseline_tree_confusion_matrix)
# plot_confusion_matrix(baseline_tree_confusion_matrix)

# Variation Decision Tree
# max_depth to 24 --> slight increase
# max_leaf_node: 64 too low, 128/256 no real difference
# min_samples_leaf: 2 same, 4 slight increase, 10 too high
variation_classifier = DecisionTreeClassifier(max_depth=12,
											  min_samples_leaf=4,
											  max_leaf_nodes=128,
											  criterion='gini')
variation_classifier_fit = variation_classifier.fit(x_train, y_train)
variation_tree_prediction = variation_classifier_fit.predict(x_val)
variation_tree_confusion_matrix = confusion_matrix(y_val, variation_tree_prediction)
print('Variation Confusion Matrix: Acc={:0.3f}%'.format(100*np.trace(variation_tree_confusion_matrix)/np.sum(variation_tree_confusion_matrix)))
print(variation_tree_confusion_matrix)
# plot_confusion_matrix(variation_tree_confusion_matrix)

# Hand-Crafted Decision Tree
# Guessing we extend our input array to be YYY x 748 + x, where x is # of additional features
# Average pixel value for each image
train_pixden = np.reshape(np.asarray([np.mean(img) for img in x_train]), (len(x_train), 1))
x_train_w_pixden = np.append(x_train, train_pixden, axis=1)

val_pixden = np.reshape(np.asarray([np.mean(img) for img in x_val]), (len(x_val), 1))
x_val_w_pixden = np.append(x_val, val_pixden, axis=1)

crafted_classifier = DecisionTreeClassifier()
crafted_classifier_fit = crafted_classifier.fit(x_train_w_pixden, y_train)
crafted_tree_prediction = crafted_classifier_fit.predict(x_val_w_pixden)
crafted_tree_confusion_matrix = confusion_matrix(y_val, crafted_tree_prediction)
print('Crafted Confusion Matrix: Acc={:0.3f}%'.format(100*np.trace(crafted_tree_confusion_matrix)/np.sum(crafted_tree_confusion_matrix)))
print(crafted_tree_confusion_matrix)

















