#!/usr/bin/python3

"""
	Program:	XorNN

	Utility:
		A Tensorflow Neural Network trained to classify whether an or statement is inclusive or exclusive
		WARNING! This program leverages and NVIDIA Graphics card tp speed things up. This program will be very slow on a cpu only system

	Author:			Troi Chua
	Date:			May 7, 2017
	Collaborators:
	Citations:		
		https://github.com/soerendip/Tensorflow-binary-classification
		https://keras.io/preprocessing/text/
		http://www.nltk.org/
		https://www.tensorflow.org/

"""

# imports python3 functionality for python2
from __future__ import division, print_function, absolute_import

# numpy is the most popular matrix manipulation library in python and a lot of libraries use their data structures
import numpy as np
# import the machine library
import tensorflow as tf

# a library for timing how long the program takes
import timeit

# start the timer
start = timeit.default_timer()

# these load the data into the program
train_txt = open("Data_1/word/train/fpWordRep_trainDAT.txt", "r")
test_txt = open("Data_1/word/test/fpWordRep_testDAT.txt", "r")
train_labels_txt = open("Data_1/labels/train_targets.txt", "r")
test_labels_txt = open("Data_1/labels/test_targets.txt", "r")

# these parse the data and organize them such that 1 word is in each partition of an array
train_list = train_txt.readlines()
test_list = test_txt.readlines()
train_labels_list = train_labels_txt.readlines()
test_labels_list = test_labels_txt.readlines()

# these convert the string values stored in the text file to numbers and formats them into a dataset acceptatble by TF
# -1 in reshape is for an unknown size
training_data = [np.asarray([float(n) for n in line.split()]).reshape(-1,299) for line in train_list]
testing_data = [np.asarray([float(n) for n in line.split()]).reshape(-1,299) for line in test_list]
training_labels = [np.asarray([float(n) for n in line.split()]).reshape(-1,2) for line in train_labels_list]
testing_labels = [np.asarray([float(n) for n in line.split()]).reshape(-1,2) for line in test_labels_list]


# Parameters
learning_rate   = 0.01	# literally the rate the network learns at;the rate the optimizer minimzes loss.
batch_size = 20			# the size of one batch of data that the network learns on so it may learn in increments
training_epochs = 5000	# the number of times the network is trained
display_step    = 100

# Network Parameters
n_hidden_1  = 20	# 1st hidden layer of neurons
n_hidden_2  = 20	# 2nd hidden layer of neurons
n_input     = 299	# number of words per sentence
n_classes	= 2		# inclusive = [0,1], exclusive = [1,0]


# placeholders variables to hold the set of sentences. None is for an unknown size
x = tf.placeholder(tf.float32, [None,n_input])		# for the sentences
y = tf.placeholder(tf.float32, [None,n_classes])	# for labels

# the network architecture
def network(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # the output (predicted value)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=1e-4)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=1e-4)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes],stddev=1e-4))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = network(x, weights, biases)	# predicted value

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# Session = execute the code
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())	# needed to initialize all the TF data structures

	# Training
	for epoch in range(training_epochs):
		
		avg_cost = 0	# initalize the cost
		total_batches = int(len(training_data) / batch_size)	

		# total batches is how many paritions the data is partition into
		trDat_batches = np.array_split(training_data, total_batches)
		trLab_batches = np.array_split(training_labels, total_batches)

		for i in range(total_batches):	# iterates through all the batches
			for j in range(len(trDat_batches[0])):	# This is needed to iterate through all the data in each batch
				
				batch_Dat, batch_Lab = trDat_batches[i][j], trLab_batches[i][j]	

				# Feed the network the current sentence
				_, c = sess.run([optimizer, cost], feed_dict={x:	batch_Dat,
														y:	batch_Lab})

				# Calculate the avg error of the current training iteration
				avg_cost += c / total_batches
		

		# will print the current status of training every display step
		if epoch % display_step == 0:
			print("Epoch:\t", '%04d' % (epoch), "\tcost =", "{:.9f}".format(avg_cost))
		
		

	print("\nTraining Finished!\n")


	# Testing method - Will return true if the predicted value matches the label
	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))


	
	succ = 0.0	# initialize how many accurate answers there are
	
	print("Predicted Output\t\tTargetOutput\tSuccess\n")

	# loops through the test cases
	for i in range(len(testing_data)):
		print(sess.run(pred, feed_dict={x: testing_data[i], y: testing_labels[i]}), end="\t")
		print(sess.run(y, feed_dict={x: testing_data[i], y: testing_labels[i]}), end="\t")
		out = sess.run(correct_prediction, feed_dict={x: testing_data[i], y: testing_labels[i]})
		print(out)
		print()
		if(out[0] == True):	# increment counter if model corrected correctly
			succ += 1


	print("\nAccuracy:\t{0}".format(succ / len(testing_data)))


stop = timeit.default_timer()	# stop the timer
print("\nTime elapsed: ", stop - start)


# close the textfiles
train_txt.close()
test_txt.close()
train_labels_txt.close()
test_labels_txt.close()


