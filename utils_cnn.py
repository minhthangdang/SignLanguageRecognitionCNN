import numpy as np
import pandas as pd
import tensorflow as tf
import math

def load_dataset():
	# read train dataset
	train_dataset = pd.read_csv('sign_mnist_train.csv')
	train_set_y_orig = labels = train_dataset['label'].values # train set labels
	train_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the training set
	train_set_x_orig = train_dataset.values # train set features
	# convert X to (m, n_H, n_W, n_C) where 
	# m is number of examples, n_H is height, n_W is width and n_C is number of channels
	train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], 28, 28, 1))

	# read test dataset
	test_dataset = pd.read_csv('sign_mnist_test.csv')
	test_set_y_orig = test_dataset['label'].values # test set labels
	test_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the test set
	test_set_x_orig = test_dataset.values # test set features
	# convert X to (m, n_H, n_W, n_C) where 
	# m is number of examples, n_H is height, n_W is width and n_C is number of channels
	test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], 28, 28, 1))

	classes = np.array(labels)
	classes = np.unique(classes)

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def create_placeholders(n_H0, n_W0, n_C0, n_y):
	"""
	Creates the placeholders for the tensorflow session.

	Arguments:
	n_H0 -- scalar, height of an input image
	n_W0 -- scalar, width of an input image
	n_C0 -- scalar, number of channels of the input
	n_y -- scalar, number of classes
	    
	Returns:
	X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
	Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
	"""

	X = tf.placeholder(dtype=tf.float32, shape=[None, n_H0, n_W0, n_C0], name='X')
	Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name='Y')

	return X, Y

def initialize_parameters():
	"""
	Initializes weight parameters to build a neural network with tensorflow. The shapes are:
	                    W1 : [3, 3, 1, 6]
	                    W2 : [5, 5, 6, 16]
	Returns:
	parameters -- a dictionary of tensors containing W1, W2
	"""
    
	W1 = tf.get_variable(name="W1", shape=[3, 3, 1, 6], initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable(name="W2", shape=[5, 5, 6, 16], initializer = tf.contrib.layers.xavier_initializer())

	parameters = {"W1": W1,
	              "W2": W2}

	return parameters

def forward_propagation(X, parameters):
	"""
	Implements the forward propagation for the model:
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing your parameters "W1", "W2"
	              the shapes are given in initialize_parameters

	Returns:
	Z3 -- the output of the last LINEAR unit
	"""

	# Retrieve the parameters from the dictionary "parameters" 
	W1 = parameters['W1']
	W2 = parameters['W2']

	# CONV2D: stride of 1, padding 'SAME'
	Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
	# RELU
	A1 = tf.nn.relu(Z1)
	# MAXPOOL: window 2x2, stride 2, padding 'VALID'
	P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
	# CONV2D: filters W2, stride 1, padding 'VALID'
	Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding = 'VALID')
	# RELU
	A2 = tf.nn.relu(Z2)
	# MAXPOOL: window 2x2, stride 2, padding 'VALID'
	P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
	# FLATTEN
	F = tf.contrib.layers.flatten(P2)
	# FULLY-CONNECTED without non-linear activation function (not not call softmax).
	# 24 neurons in output layer.
	Z3 = tf.contrib.layers.fully_connected(F, 120, activation_fn=None)
	Z4 = tf.contrib.layers.fully_connected(Z3, 84, activation_fn=None)
	Z5 = tf.contrib.layers.fully_connected(Z4, 24, activation_fn=None)

	return Z5

def compute_cost(Z3, Y):
	"""
	Computes the cost

	Arguments:
	Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
	Y -- "true" labels vector placeholder, same shape as Z3

	Returns:
	cost - Tensor of the cost function
	"""

	# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

	return cost

def random_mini_batches(X, Y, mini_batch_size = 64):
	"""
	Creates a list of random minibatches from (X, Y)

	Arguments:
	X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
	Y -- true "label" vector of shape (m, n_y)
	mini_batch_size - size of the mini-batches, integer

	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""

	m = X.shape[0]                  # number of training examples
	mini_batches = []

	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches