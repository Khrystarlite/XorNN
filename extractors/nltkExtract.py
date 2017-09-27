#!/usr/bin/python3

from nltk.corpus import *


rural = abc.sents('rural.txt')
sents_list = [" ".join(sent) for sent in rural]
ortxt = []
for i in sents_list:
	if ' or ' in i:
		ortxt.append(i)

f = open("Data_1/raw/0_rural_orDB.txt","w")
for i in ortxt:
	f.write(i + "\n")
f.close()

print(len(ortxt))


emma = gutenberg.sents('austen-emma.txt')
sents_list = [" ".join(sent) for sent in emma]
ortxt = []
for i in sents_list:
	if ' or ' in i:
		ortxt.append(i)

f = open("Data_1/raw/1_austenEma_orDB.txt","w")
for i in ortxt:
	f.write(i + "\n")
f.close()

print(len(ortxt))


sense = gutenberg.sents('austen-sense.txt')
sents_list = [" ".join(sent) for sent in sense]
ortxt = []
for i in sents_list:
	if ' or ' in i:
		ortxt.append(i)

f = open("Data_1/raw/2_austenSense_orDB.txt","w")
for i in ortxt:
	f.write(i + "\n")
f.close()

print(len(ortxt))


bear = gutenberg.sents('burgess-busterbrown.txt')
sents_list = [" ".join(sent) for sent in bear]
ortxt = []
for i in sents_list:
	if ' or ' in i:
		ortxt.append(i)

f = open("Data_1/raw/3_burgessBrown_orDB.txt","w")
for i in ortxt:
	f.write(i + "\n")
f.close()

print(len(ortxt))


paradise = gutenberg.sents('milton-paradise.txt')
sents_list = [" ".join(sent) for sent in paradise]
ortxt = []
for i in sents_list:
	if ' or ' in i:
		ortxt.append(i)


f = open("Data_1/raw/4_miltonParadise_orDB.txt","w")
for i in ortxt:
	f.write(i + "\n")
f.close()

print(len(ortxt))

