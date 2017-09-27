#!/usr/bin/python3

from gensim.models import Word2Vec
from progressbar import ProgressBar
from time import sleep
from keras.preprocessing import text

import pandas as pd
import keras
import numpy as np
import tensorflow as tf


train_txt = open("../Data_1/train.txt","r")
test_txt = open("../Data_1/test.txt","r")
train_labels_txt = open("../Data_1/labels/train_targets.txt", 'r')
test_labels_txt = open("../Data_1/labels/test_targets.txt", 'r')

train_list = train_txt.readlines()
test_list = test_txt.readlines()
train_labels_list = train_labels_txt.readlines()
test_labels_list = test_labels_txt.readlines()

training_data = []
testing_data = []
training_labels = []
testing_labels = []



for sentence in train_list:
	buff = keras.preprocessing.text.one_hot(sentence,n=1/50000,lower=False,filters='\t\n')
	training_data.append(buff)

for sentence in test_list:
	buff = keras.preprocessing.text.one_hot(sentence,n=1/50000,lower=False,filters='\t\n')
	testing_data.append(buff)



for label in train_labels_list:
	training_labels.append(int(label))

for label in test_labels_list:
	testing_labels.append(int(label))

max_len = 0
# padding = float('-inf')
padding = -1.0
for i in training_data:
	if(len(i) > max_len):
		max_len = len(i)

for i in testing_data:
	if(len(i) > max_len):
		max_len = len(i)

for i in training_data:
	while(len(i) < max_len):
		i.append(padding)

for i in testing_data:
	while(len(i) < max_len):
		i.append(padding)

f = open('../Data_1/word/train/fpWordRep_trainDAT.txt', 'w')
for i in training_data:
	for j in i:
		f.write(str(j) + " ")
	f.write("\n")

f.close()
f = open('../Data_1/word/test/fpWordRep_testDAT.txt', 'w')
for i in testing_data:
	for j in i:
		f.write(str(j) + " ")
	f.write("\n")
f.close()
