from flask import Flask,request,jsonify
import pickle
from flask_restful import reqparse
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import string
from nltk import pos_tag

app = Flask(__name__)

@app.route('/')
def hello():
	return 'Up and running'

@app.route('/predict',methods = ['POST'])
def predict():
	clean_words = []
	current_features = {}
	#Importing Model & getting input
	with open('MNBmodel','rb') as w:
		imp_model = pickle.load(w)
	json = request.get_json()
	lst = list(json[0].values())
	review = lst[0]

	lemmatizer= WordNetLemmatizer()
	stop=stopwords.words("english")
	punctuations = list(string.punctuation)
	stop=stop + punctuations
	#Stop words list
	stop
	words = review.split()
	for w in words :
		if w.lower() not in stop :
			pos = pos_tag(w)
			clean_words=lemmatizer.lemmatize(w,pos= get_simple_pos(pos[0][1]))#[0]will give tuple #[1] will be noun
			output_words.append(clean_words.lower())
	words_set=set(clean_words)
	for w in features:
		current_features[w]=w in words_set
	x = current_features

	prediction = imp_model.classify(review)
	print("here:",prediction)
	if(prediction[0]==1):
		resp = 'Positive'
	else:
		resp = 'Negative'
	return jsonify({'prediction': resp})

if __name__=='__main__':
	#app.run(host='192.168.0.100', port='8000', debug=True)
	app.run(host='0.0.0.0',)
