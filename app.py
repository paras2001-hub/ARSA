import json
from flask import Flask,request,jsonify
import pickle
from flask_restful import reqparse
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
import os
import re
import shutil
import tensorflow as tf
from collections import Counter
import numpy as np
import sklearn
import tensorflow_hub as hub
from tensorflow_hub import KerasLayer
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route('/')
def hello():
	return 'Up and running'

@app.route('/predict',methods = ['POST'])
def predict():
	output_words = []
	current_features = {}
	json_ = request.get_json()
	print(json_)
	lst = list(json_[0].values())
	with open('tokenizer.json') as json_file:
		print(json_file)
		json_string = json.load(json_file)
	tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
	review = lst[0]
	opt = lst[1]

	lemmatizer= WordNetLemmatizer()

	stop= ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ai','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't",'!','"','#','$','%','&',"'",'(',')','*','+',',','-','.',',',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']

	words = review.split()
	for w in words :
		if w.lower() not in stop :
			if w.lower().startswith("J"):
				pos= wordnet.ADJ
			elif w.lower().startswith("V"):
				pos= wordnet.VERB
			elif w.lower().startswith("N"):
				pos= wordnet.NOUN
			elif w.lower().startswith("R"):
				pos= wordnet.ADV
			else:
				pos= wordnet.NOUN

			clean_words=lemmatizer.lemmatize(w,pos)           
			output_words.append(clean_words.lower())
	features = []
	if opt == 0:
		with open('sample words unigram.txt', 'r') as filehandle:
			filecontents = filehandle.readlines()
		for line in filecontents:
			current_place = line[:-1]
			features.append(current_place)
		words_set = set(output_words)
		for w in features:
			current_features[w]=w in words_set
		x = current_features
		with open('NBmodel','rb') as w:
			imp_model1 = pickle.load(w)
		prediction = imp_model1.classify(x)
	
	elif opt == 1:
		with open('count_vec_uni_model','rb') as w:
			cv_uni = pickle.load(w)
		x_paragraph=[" ".join(output_words)]
		x=cv_uni.transform(x_paragraph)
		with open('DT_CV_UNI_model','rb') as w:
			imp_model2 = pickle.load(w)
		prediction = imp_model2.predict(x)

	elif opt == 2:
		with open('count_vec_bi_model','rb') as w:
			cv_bi = pickle.load(w)
		x_paragraph=[" ".join(output_words)]
		x=cv_bi.transform(x_paragraph)
		with open('DT_CV_BI_model','rb') as w:
			imp_model3 = pickle.load(w)
		prediction = imp_model3.predict(x)

	elif opt == 3:
		with open('count_vec_uni_model','rb') as w:
			cv_uni = pickle.load(w)
		x_paragraph=[" ".join(output_words)]
		x=cv_uni.transform(x_paragraph)
		with open('SVM_CV_UNI_model','rb') as w:
			imp_model4 = pickle.load(w)
		prediction = imp_model4.predict(x)

	elif opt == 4:
		with open('count_vec_bi_model','rb') as w:
			cv_bi = pickle.load(w)
		x_paragraph=[" ".join(output_words)]
		x=cv_bi.transform(x_paragraph)
		with open('SVM_CV_BI_model','rb') as w:
			imp_model5 = pickle.load(w)
		prediction = imp_model5.predict(x)

	elif opt == 5:
		with open('CV_MNB_UNI','rb') as w:
			cv_uni = pickle.load(w)
		x_paragraph=[" ".join(output_words)]
		x=cv_uni.transform(x_paragraph)
		with open('MNB_CV_UNI_model','rb') as w:
			imp_model6 = pickle.load(w)
		prediction = imp_model6.predict(x)

	elif opt == 6:
		with open('CV_MNB_BI','rb') as w:
			cv_bi = pickle.load(w)
		x_paragraph=[" ".join(output_words)]
		x=cv_bi.transform(x_paragraph)
		with open('MNB_CV_BI_model','rb') as w:
			imp_model7 = pickle.load(w)
		prediction = imp_model7.predict(x)
	
	elif opt == 7:
		max_features =50000
		embedding_dim =16
		sequence_length = 50
		new_model = tf.keras.Sequential()
		new_model.add(tf.keras.layers.Embedding(max_features +1, embedding_dim, input_length=sequence_length, embeddings_regularizer = regularizers.l2(0.005))) 
		new_model.add(tf.keras.layers.Dropout(0.4))
		new_model.add(tf.keras.layers.LSTM(embedding_dim,dropout=0.2, recurrent_dropout=0.2,return_sequences=True,kernel_regularizer=regularizers.l2(0.005), bias_regularizer=regularizers.l2(0.005)))
		new_model.add(tf.keras.layers.Flatten())
		new_model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001),bias_regularizer=regularizers.l2(0.001),))
		new_model.add(tf.keras.layers.Dropout(0.4))
		new_model.add(tf.keras.layers.Dense(8, activation='relu',kernel_regularizer=regularizers.l2(0.001),bias_regularizer=regularizers.l2(0.001),))
		new_model.add(tf.keras.layers.Dropout(0.4))
		new_model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

		new_model.load_weights('tf_lstmmodel_weights.h5')
		rev_wordlist = []
		review_token = ((tokenizer.texts_to_sequences(output_words)))
		for i in review_token:
			for j in i:
				rev_wordlist.append(j)
		sequence_length=50
		pad_review = pad_sequences([rev_wordlist], padding='post', maxlen=sequence_length)
		res = new_model.predict(pad_review)
		if res[0][0] >= 0.805:
			prediction = "pos"
		else:
			prediction = "neg" 
	elif opt == 8:
		model_NN = tf.keras.models.load_model('Neural_Network', compile = False, custom_objects={"KerasLayer": KerasLayer})
		res = model_NN.predict(output_words)
		
		if res[0][0] > res[0][1]:
			prediction = "pos"
		else:
			prediction = "neg"
			
	print("here:",prediction)
	if(prediction=="pos"):
		resp = 'Positive'
	else:
		resp = 'Negative'
	return jsonify({'prediction': resp})

if __name__=='__main__':
	#app.run(host='192.168.0.179', port='5000', debug=True)
	app.run(host='0.0.0.0',)
