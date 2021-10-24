#!/usr/bin/python
# -*- coding: latin-1 -*-

"""
Algorithm using a CNN to classify pixels into the following two classes: DNA (0) and non-DNA(1)
--> CNN with 2 convolutional layer, 2 pooling layer and 1 fully connected layer

To segment contrast-normalized and rescaled tile set images.

date: 12.02.16
author: Fabienne K. Roessler
"""

# Import necessary libraries
from __future__ import print_function
import numpy as np
import os
from scipy import ndimage
import imageio
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from PIL import Image
import scipy.misc
import math
import random as rd
from datetime import datetime


# ATTENTION!
# New user: Define the following variables before first use
# ATTENTION!
# Define path to input folder containing contrast-normalized and rescaled images:
input_path = ...
# Define path to output folder (segmented images will be saved there):
output_path = ...
# Define path to checkpoint file of CNN (needs to look like this: path + 'trained_PixelClassifier.ckpt'):
save_path = ...
# Define size of rescaled images in pixel (!!! Always one pixel bigger than actual size !!!):
whole_image_size = ...							# e.g. whole_image_size = 1025


# Definition of rest of constants (!!! DO NOT CHANGE THEM !!!)
image_size = 28                                 # Size of smaller image which is given to the classifier

pixel_depth = int(255)                          # Max possible value of a pixel
num_labels = 2                                  # 2 classes
num_channels = 1                                # Grayscale (needed for CNNs)

batch_size = (whole_image_size-image_size)		# Size of minibatch, which is handed over to the CNN during each step of training

patch_sizec1 = 7                                # Kernel size of first convolutional layer
patch_sizec2 = 3 								# Kernel size of second convolutional layer
depth = 12										# Depth of first convolutional layer
depth_2 = 24									# Depth of second convolutional layer
num_fullycon = 36                               # Number of neurons in the fully connected layer

np.random.seed(42)


# Get list of tile set folders saved in input folder and create them under the same name in output folder
squares = os.listdir(input_path)
for m in range(len(squares)):
	newFolder = os.path.join(output_path, squares[m])
	if not os.path.exists(newFolder):
		os.makedirs(newFolder)

# Get list of tile set folders in output folder
outputSquares = os.listdir(output_path)


# Function to load an image and convert it into a numpy array
def load_image(file):
	im = Image.open(file)
	im_array = np.array(im).astype(float)
	return im_array

# Function to create the dataset and the corresponding label array
def create_sliced_image_dataset(image, y, size):
	dataset = np.ndarray(shape = (size, image_size, image_size), dtype = np.float32)

	for x in range(size):
		dataset[x, :, :] = image[y:(y+image_size), x:(x+image_size)]

	return dataset

# Function to reformat the dataset array and label array in the desired shape
def reformat(dataset):
	dataset = dataset.reshape(
		(-1, image_size, image_size, num_channels)).astype(np.float32)
	return dataset

# Function to initialize and run trained CNN
def start_CNN(input):
	with tf.device("/gpu:0"):
		graph = tf.Graph()
	
		with graph.as_default():
			tf_image = tf.placeholder(
				tf.float32, shape=(batch_size, image_size, image_size, num_channels))
		
			# Variables (weights and biases for each layer)
			layer1_weights = tf.Variable(tf.truncated_normal(
				[patch_sizec1, patch_sizec1, num_channels, depth], stddev=0.1))
			layer1_biases = tf.Variable(tf.zeros([depth]))	

			layer2_weights = tf.Variable(tf.truncated_normal(
				[patch_sizec2, patch_sizec2, depth, depth_2], stddev=0.1))
			layer2_biases = tf.Variable(tf.zeros([depth_2]))
		
			layer3_weights = tf.Variable(tf.truncated_normal(
				[int(math.ceil(image_size/4.0)) * int(math.ceil(image_size/4.0)) * depth_2, num_fullycon], stddev=0.1))
			layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_fullycon]))
		
			layer4_weights = tf.Variable(tf.truncated_normal(
				[num_fullycon, num_labels], stddev=0.1))
			layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

			# Define the CNN model
			def model(data):
				conv1 =  tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
				hidden1 = tf.nn.relu(conv1 + layer1_biases)
				pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			
				conv2 =  tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
				hidden2 = tf.nn.relu(conv2 + layer2_biases)
				pool2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			
				pool2_flat = tf.reshape(pool2, [-1, int(math.ceil(image_size/4.0)) * int(math.ceil(
					image_size/4.0)) * depth_2])
				
				fullycon1 = tf.nn.relu(tf.matmul(pool2_flat, layer3_weights) + layer3_biases)
				
				return tf.matmul(fullycon1, layer4_weights) + layer4_biases

			# Predictions for the training, validation and test dataset
			image_prediction = tf.nn.softmax(model(tf_image))
			
			# Saver of the trained model
			saver = tf.train.Saver()


			# Where the magic actually happens ;)
			with tf.Session(graph=graph) as session:
				start_time = datetime.now()
				saver.restore(session, save_path)
				print('Initialized')	

				for image in images:
					print('Image to segment: ', image)
					image_path = os.path.join(input, image)
					im = load_image(image_path)
					mean = np.mean(im)
					im = (im-mean)	

					im_segment = np.ndarray(shape = (batch_size, batch_size),
						dtype = np.uint8)

					for y in range(batch_size):
						print(y)

						batch_dataset = create_sliced_image_dataset(
							im, y, batch_size)
						batch_dataset = reformat(batch_dataset)
						
						print('Batch set ', batch_dataset.shape)

						feed_dict = {tf_image : batch_dataset}
						predictions = session.run(image_prediction,
							feed_dict=feed_dict)
	
						for p in range(batch_size):
							if predictions[p][1] > predictions[p][0]:
								im_segment[y, p] = int(255)
							else:
								im_segment[y, p] = int(0)
	
					new_name = image.replace('.tif', '_segmented.tif')
					#scipy.misc.imsave(output+new_name, im_segment)
					imageio.imwrite(output+new_name, im_segment)
				
					print('Image saved as: ', new_name)
					print('Time taken: ', datetime.now() - start_time)

				print('Finished!')


# Main part segmenting images into DNA and non-DNA parts using a trained CNN
# Loop over tile sets
for s in range(len(squares)):
	print(squares[s])
	for o in range(len(outputSquares)):
		if squares[s] == outputSquares[o]:
			output = output_path+outputSquares[o]+'/'
			print(output)
			currentInput = input_path+squares[s]
			print(currentInput)
			# Get current image in tile set
			images = os.listdir(currentInput)
			# Run CNN on current image
			start_CNN(currentInput)