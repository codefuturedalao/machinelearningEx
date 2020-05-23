import sys
import pandas as pd
import numpy as np
import re
import os
import math
import scipy.sparse as sp
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

cateNum = 20
trainDataNumPerCate = 250 
studyRate = 1
K = 10 

class Perception:
	def __init__(self,trainFilePath,testFilePath):
		self.DataNumPerCate = [390,390,380,390,390,390,390,250,390,390,390,310,380,390,390,390,390,370,310,360]
		self.trainLabels,self.trainLabel,self.trainData = self.readData(trainFilePath)
		self.testLabels,self.testLabel,self.testData = self.readData(testFilePath)

	def readData(self,filePath):
		data = []
		originFileDir = os.getcwd()
		os.chdir(filePath)
		fileChdir = os.getcwd()
		labels = os.listdir(fileChdir)
		trainLabels = []
		fileList = []
		j = 0
		for di in labels:  #pass every subdir in the trainData
			i = 0
			for root,dirs,files in os.walk(di):
				for fi in files:
#					i += 1
					
					if i < self.DataNumPerCate[j]:
						fileList.append(fi)
						trainLabels.append(di)
					i = i + 1
			j += 1
					
		i = 0	
		for fi in fileList:
			fd = open(trainLabels[i] +'/' +  fi)
			#remove the email header
			msg = Parser().parse(fd)
			for part in msg.walk():
				emailBody = part.get_payload(decode=True)	
			data.append(emailBody)
			i = i + 1
		#change to the origin dir
		os.chdir(originFileDir)

		return trainLabels,labels,data

	def preprocessData(self,data):
		# change all the text to the lowe case. and remove all the punctuation
		rule = re.compile("[^a-zA-Z\d ]")
		for i in range(len(data)):
			data[i] = re.sub("[^a-zA-Z\d ]",'',data[i].lower())	

		data = [word_tokenize(entry) for entry in data]
		# remove stop word, Non-Numeric and perfor Word Stemming/Lemmenting
		tag_map = defaultdict(lambda : wn.NOUN)
		tag_map['J'] = wn.ADJ
		tag_map['V'] = wn.VERB
		tag_map['R'] = wn.ADV
		i = 0
		for index,entry in enumerate(data):
			Final_words = []
			word_Lemmatized = WordNetLemmatizer()
			for word,tag in pos_tag(entry):
				if word not in stopwords.words('english') and word.isalpha():
					word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
					Final_words.append(word_Final)
#			if i < 5:
#				print(word_Final)
			data[i] = Final_words
			i = i + 1
		return data

	def getWordList(self,data):
		n = len(data)
		word_dict = [] 
		for i in range(n):
			dat = list(set(data[i]))
			for item in dat:
				word_dict.append(item)	
		return word_dict

	def train(self,data,word_list):
		self.predictor = []
		train_X = []
		m = len(word_list)
		n = len(data)
		#initialize
		for i in range(cateNum):
			self.predictor.append([np.zeros((m,1)),0]) 
		#print self.predictor
		for i in range(n/K):
			X = np.zeros((1,m),dtype=np.int32)
			for j in range(i*K,i*K+K):
				for word in data[j]:
					if word in word_list:
						index = word_list.index(word)
						X[0,index] += 1
			x = sp.csr_matrix(X)
			train_X.append(x)
		#print train_X
		#print n
		maxBoundary = 0
		minBoundary = -self.DataNumPerCate[0] / K
		for i in range(cateNum):
			maxBoundary += self.DataNumPerCate[i] / K
			minBoundary += self.DataNumPerCate[i] / K
			trainSequence =[]
			trainNum = 0
			totalNum = 0
			flag = 0
			while flag == 0:
				flag = 1
				for j in range(n/K):
					train_y = 1
					totalNum += 1
					if j >= minBoundary and j < maxBoundary:
						train_y = 1
					else:
						train_y = -1
					target_y = train_X[j].dot(self.predictor[i][0]) + self.predictor[i][1]
		#			if j < 0:
		#				print('train_X is\n',train_X[j])
		#				print('self.predictor[i][0] is',self.predictor[i][0])
		#				print('self.predictor[i][1] is',self.predictor[i][1])
		#				print('target_y is\n',target_y)
		#				print('train_y is\n',train_y)
		#				print target_y * train_y
					if target_y * train_y <= 0:
						flag = 0
		#				trainNum += 1
		#				trainSequence.append(1)	
						self.predictor[i][0] += studyRate * train_y * train_X[j].T
						self.predictor[i][1] += studyRate * train_y 	
		#			else:
		#				trainSequence.append(0)
		#			if j < 0:
		#				print('self.predictor[i][0] is',self.predictor[i][0])
		#				print('self.predictor[i][1] is',self.predictor[i][1])
	#			print('totalNum is %',totalNum,'trainNum is %',trainNum)
	#			print trainSequence
		
	def predict(self,samples,word_list):
		m = len(word_list)
		ret = []
		for sam in samples:
			X = np.zeros((1,m),dtype=np.int32)
			for word in sam:
				if word in word_list: 
					index = word_list.index(word)
					X[0,index] += 1
			prop = []		
			for i in range(cateNum):
				target_y = np.dot(X,self.predictor[i][0]) + self.predictor[i][1]
				prop.append(target_y)
			index = prop.index(max(prop))
	#		print X
	#		print prop
	#		print index
			ret.append(self.trainLabel[index])
		return ret
	
		
def main():
	p = Perception("./20news-bydate/20news-bydate-train/","./20news-bydate/20news-bydate-test/")
	p.trainData = p.preprocessData(p.trainData)
	#print p.trainData 
	p.testData = p.preprocessData(p.testData)
	#print p.testData
	word_list = p.getWordList(p.trainData)
	#print len(word_list)
	p.train(p.trainData,word_list)
	ret = p.predict(p.trainData,word_list)
	#print ret
	print(classification_report(p.testLabels,ret))	
	print ret.count(p.trainLabel[cateNum-1])
	#data = ['dog cat animal mew cat','computer electronic keyborad','tea water drink cat']
	#data = [['dog','cat','animal','mew','cat'],['computer','electronic','keyborad'],['tea','water','drink','cat']]
	#data = p.preprocessData(data)
	#print data
	#word_list = p.getWordList(data)
	#print word_list
	#p.train(data,word_list)
	#ret = p.predict(data,word_list)
	
	

if __name__ == "__main__":
	main()
