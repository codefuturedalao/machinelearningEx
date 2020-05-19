import sys
import pandas as pd
import numpy as np
import re
import os
from email.parser import Parser
from sys import argv
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score

trainDataNum = 20  #num per class in traindata

class NaiveBayes:
	def __init__(self,trainFile):
		self.trainFile = trainFile
		self.labels,self.trainData = self.readData(trainFile)
		print self.labels
		

	def readData(self,filePath):
		data = []
		os.chdir(filePath)
		fileChdir = os.getcwd()
		labels = os.listdir(fileChdir)
		fileList = []
		for di in labels:  #pass every subdir in the trainData
			i = 0
			for root,dirs,files in os.walk(di):
				for fi in files:
					if i < trainDataNum:
						fileList.append(fi)
					i = i + 1
		i = 0	
		for fi in fileList:
			fd = open(labels[i/trainDataNum] +'/' +  fi)
			msg = Parser().parse(fd)
			for part in msg.walk():
				emailBody = part.get_payload(decode=True)	
			data.append(emailBody)
			i = i + 1

		return labels,data
	def preprocessData(self):
		# remove the blank line
		# todo
		# change all the text to the lowe case. and remove all the punctuation
		rule = re.compile("[^a-zA-Z\d ]")
		for i in range(len(self.trainData)):
			self.trainData[i] = re.sub("[^a-zA-Z\d ]",'',self.trainData[i].lower())	

		self.trainData = [word_tokenize(entry) for entry in self.trainData]
		# remove stop word, Non-Numeric and perfor Word Stemming/Lemmenting
		tag_map = defaultdict(lambda : wn.NOUN)
		tag_map['J'] = wn.ADJ
		tag_map['V'] = wn.VERB
		tag_map['R'] = wn.ADV
		i = 0
		for index,entry in enumerate(self.trainData):
			Final_words = []
			word_Lemmatized = WordNetLemmatizer()
			for word,tag in pos_tag(entry):
				if word not in stopwords.words('english') and word.isalpha():
					word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
					Final_words.append(word_Final)
#			if i < 5:
#				print(word_Final)
			self.trainData[i] = Final_words
			i = i + 1
#		postemp = pos_tag(['women','lines'])
#		word = postemp[0][0]
#		tag = postemp[0][1]
#		wordTemp = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#		print(postemp)
#		print(word)
#		print(tag)
#		print(wordTemp)
	
	def baseProb(self):
		basePro = []
		for i in self.labels:
			basePro.append(1.0/len(self.labels))
		print basePro
		return basePro
		

#def main(trainFiles,testFile):
def main():
	naiveB = NaiveBayes("./20news-bydate/20news-bydate-train/")
	naiveB.preprocessData()
	naiveB.baseProb()
	
	


	print(naiveB.trainData[:3])

if __name__ == "__main__":
#	if len(argv) != 3:
#		print "usage python %s trainFile(in) testFile(out)" %__file__
#		sys.exit(0)
#	main(argv[1],argv[2])
	main()
