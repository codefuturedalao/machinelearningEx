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

trainDataNum = 300  #num per class in traindata

class NaiveBayes:
	def __init__(self,trainFile):
		self.trainFile = trainFile
		self.labels,self.trainData = self.readData(trainFile)
		

	def readData(self,filePath):
		data = []
		os.chdir(filePath)
		fileChdir = os.getcwd()
		labels = os.listdir(fileChdir)
		finalLabels = []
		fileList = []
		for di in labels:  #pass every subdir in the trainData
			i = 0
			for root,dirs,files in os.walk(di):
				for fi in files:
					if i < trainDataNum:
						fileList.append(fi)
						finalLabels.append(di)
					i = i + 1
		i = 0	
		for fi in fileList:
			fd = open(labels[i/trainDataNum] +'/' +  fi)
			msg = Parser().parse(fd)
			for part in msg.walk():
				emailBody = part.get_payload(decode=True)	
			data.append(emailBody)
			i = i + 1

		return finalLabels,data
	def preprocessData(self):
		d = {'text_final':self.trainData,'labels':self.labels}
		data = pd.DataFrame(data=d)
		# remove the blank line
		data['text_final'].dropna(inplace=True)	
		# todo
		# change all the text to the lowe case. and remove all the punctuation
		rule = re.compile("[^a-zA-Z\d ]")
		for i in range(data.size/2):
			data.loc[i,'text_final'] = re.sub("[^a-zA-Z\d ]",'',data.loc[i,'text_final'].lower())

		data['text_final'] = [word_tokenize(entry) for entry in data['text_final']]
		# remove stop word, Non-Numeric and perfor Word Stemming/Lemmenting
		tag_map = defaultdict(lambda : wn.NOUN)
		tag_map['J'] = wn.ADJ
		tag_map['V'] = wn.VERB
		tag_map['R'] = wn.ADV
		for index,entry in enumerate(data['text_final']):
			Final_words = []
			word_Lemmatized = WordNetLemmatizer()
			for word,tag in pos_tag(entry):
				if word not in stopwords.words('english') and word.isalpha():
					word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
					Final_words.append(word_Final)
#			if i < 5:
#				print(word_Final)
			data.loc[index,'text_final'] = str(Final_words)

		self.trainData = data
	
	def process(self):
		Train_X,Test_X,Train_Y,Test_Y = model_selection.train_test_split(self.trainData['text_final'],self.trainData['labels'],test_size=0.3)
		#print(Train_Y)
		Encoder = LabelEncoder()
		Train_Y = Encoder.fit_transform(Train_Y)
		Test_Y = Encoder.fit_transform(Test_Y)
		#print(Train_X)
		#print(Train_Y)
		Tfidf_vect = TfidfVectorizer(max_features=5000)
		Tfidf_vect.fit(self.trainData['text_final'])
		#print(Tfidf_vect.vocabulary)
		Train_X_Tfidf = Tfidf_vect.transform(Train_X)
		Test_X_Tfidf = Tfidf_vect.transform(Test_X)
		#print(Train_X_Tfidf)
		#print(Test_X_Tfidf)
		Naive = naive_bayes.MultinomialNB()
		Naive.fit(Train_X_Tfidf,Train_Y)
		
		predictions_NB = Naive.predict(Test_X_Tfidf)

		print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB,Test_Y) * 100)

	def baseProb(self):
		basePro = []
		for i in self.labels:
			basePro.append(1.0/len(self.labels))
		print basePro
		return basePro
	
	def word_prob(self):
		n = len(self.trainData)
		#create dic
		word_dict = {} 
		i = 0	
		for i in range(n):
			dat = list(set(self.trainData[i]))
			
			for word in dat:
				if word not in word_dict:
					word_dict[word] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
				word_dict[word][i/trainDataNum] += 1
		print word_dict
		#for i in word_dict:
#			dt = word_dict[i]
#			word_dict[i] = 
			
		

#def main(trainFiles,testFile):
def main():
	naiveB = NaiveBayes("./20news-bydate/20news-bydate-train/")
	naiveB.preprocessData()
	naiveB.process()
	
	



if __name__ == "__main__":
#	if len(argv) != 3:
#		print "usage python %s trainFile(in) testFile(out)" %__file__
#		sys.exit(0)
#	main(argv[1],argv[2])
	main()
