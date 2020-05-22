import sys
import pandas as pd
import numpy as np
import re
import os
import math
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
from sklearn.metrics import classification_report 

trainDataNum = 300  #num per class in traindata

class NaiveBayes:
	def __init__(self,trainFile):
		self.trainFile = trainFile
		self.trainLabels,self.labels,self.trainData = self.readData(trainFile)
		#print self.labels
		

	def readData(self,filePath):
		data = []
		originFileDir = os.getcwd()
		os.chdir(filePath)
		fileChdir = os.getcwd()
		labels = os.listdir(fileChdir)
		trainLabels = []
		fileList = []
		for di in labels:  #pass every subdir in the trainData
			i = 0
			for root,dirs,files in os.walk(di):
				for fi in files:
					fileList.append(fi)
					trainLabels.append(di)
		i = 0	
		for fi in fileList:
			fd = open(trainLabels[i] +'/' +  fi)
			msg = Parser().parse(fd)
			for part in msg.walk():
				emailBody = part.get_payload(decode=True)	
			data.append(emailBody)
			i = i + 1
		#change to the origin dir
		os.chdir(originFileDir)

		return trainLabels,labels,data
	def preprocessData(self):
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
	
	def baseProb(self,labels):	
		basePro = [i for i  in range(20)]
		for i in labels:
			index = self.labels.index(i)
			basePro[index] += 1
		basePro = [i/float(len(labels)) for i in basePro]
		#print basePro
		return basePro
	
	def word_prob(self,data,labels):
		n = len(data)
		#create dic
		word_dict = {} 
		i = 0	
		for i in range(n):
			lab = labels[i]
			dat = list(set(data[i]))
			
			for word in dat:
				if word not in word_dict:
					word_dict[word] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
				index = self.labels.index(lab)
				word_dict[word][index] += 1
		sum_array = []
		for i in range(20):
			sum_temp = 0
			for key,values in word_dict.items():
				sum_temp += values[i]		
			sum_array.append(sum_temp)
		for i in word_dict:
			dt = word_dict[i]
			for j in range(20):
				word_dict[i][j] = word_dict[i][j] / (float(sum_array[j]) + len(word_dict)) #multinomial
		return word_dict
		
	#	print word_dict[i]	
	def predict(self,samples,word_prob,base_p):
		prop = [i for i in range(20)]
		ret = []
		for sam in samples:
			for i in range(20):
				prop[i] = math.log(base_p[i])
			for word in sam:
				if word not in word_prob:
					continue
				for j in range(20):
					prop[j] += math.log(word_prob[word][j])
			index = prop.index(max(prop)) 
			ret.append(self.labels[index])
		return ret
				
		

#def main(trainFiles,testFile):
def main():
	naiveB = NaiveBayes("./20news-bydate/20news-bydate-train/")
	naiveB.preprocessData()
	testNaiveB = NaiveBayes("./20news-bydate/20news-bydate-test/")
	testNaiveB.preprocessData()

	Base_p = naiveB.baseProb(naiveB.trainLabels)
	word_dt = naiveB.word_prob(naiveB.trainData,naiveB.trainLabels)
	ret = naiveB.predict(testNaiveB.trainData,word_dt,Base_p)	
	print(classification_report(testNaiveB.trainLabels,ret))	


#	print(naiveB.trainData[:3])

if __name__ == "__main__":
#	if len(argv) != 3:
#		print "usage python %s trainFile(in) testFile(out)" %__file__
#		sys.exit(0)
#	main(argv[1],argv[2])
	main()
