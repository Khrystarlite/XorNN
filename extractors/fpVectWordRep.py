#!/usr/bin/python3

from gensim.models import Word2Vec
from progressbar import ProgressBar
from time import sleep

import pandas as pd
import numpy as np
import tensorflow as tf
import os

train_txt = open("../Data_1/train.txt","r")
test_txt = open("../Data_1/test.txt","r")
train_labels_txt = open("../Data_1/labels/train_targets.txt")
test_labels_txt = open("../Data_1/labels/test_targets.txt")

train_list = train_txt.readlines()
test_list = test_txt.readlines()
train_labels_list = train_labels_txt.readlines()
test_labels_list = test_labels_txt.readlines()

train = []
test = []
training_data = []
testing_data = []
training_labels = []
testing_labels = [] 

for sentence in train_list:
	buff = []
	x = sentence.split()
	for i in x:
		buff.append(i)
	train.append(buff)

for sentence in test_list:
	buff = []
	x = sentence.split()
	for i in x:
		buff.append(i)
	test.append(buff)

max_len = 0
padding = 'Khrystarlite'

for i in train:
	if(len(i) > max_len):
		max_len = len(i)

for i in test:
	if(len(i) > max_len):
		max_len = len(i)

for i in train:
	while(len(i) < max_len):
		i.append(padding)

for i in test:
	while(len(i) < max_len):
		i.append(padding)


training_vec = Word2Vec(train, workers=4, min_count=1)
testing_vec = Word2Vec(test, workers=4, min_count=1)

for word in train:
	training_data.append(training_vec[word])

for word in test:
	testing_data.append(testing_vec[word])

for label in train_labels_list:
	training_labels.append(int(label))

for label in test_labels_list:
	testing_labels.append(int(label))



for i in range(80):
	df = training_data[i]
	np.savetxt('../Data_1/vector/train/{0}_fpVectWordRep_trainDAT.txt'.format(i+1), df, delimiter='\t')

for i in range(20):
	df = testing_data[i]
	np.savetxt('../Data_1/vector/test/{0}_fpVectWordRep_testDAT.txt'.format(i+1), df, delimiter='\t')





