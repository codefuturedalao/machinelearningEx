import sys
import pandas as pd
import numpy as np
import re
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

class NaiveBayes:
	def __init__(self,trainFile):
		self.trainFile = trainFile
		self.trainData = self.readData(trainFile)

	def readData(self,file):
		data = []
		fd = open(file)
		for line in fd.readlines():
			data.append(line)
		return data
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
		

#def main(trainFiles,testFile):
def main():
	naiveB = NaiveBayes("/home/jacksonsang/Downloads/20news-bydate/20news-bydate/20news-bydate-train/alt.atheism/51173")
	naiveB.preprocessData()
	
	


	print(naiveB.trainData[:16])

if __name__ == "__main__":
#	if len(argv) != 3:
#		print "usage python %s trainFile(in) testFile(out)" %__file__
#		sys.exit(0)
#	main(argv[1],argv[2])
	main()
