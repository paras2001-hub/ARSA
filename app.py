from flask import Flask,request,jsonify
import pickle
from flask_restful import reqparse
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag

app = Flask(__name__)

@app.route('/')
def hello():
	return 'Up and running'

@app.route('/predict',methods = ['POST'])
def predict():
	output_words = []
	current_features = {}
	json = request.get_json()
	print(json)
	lst = list(json[0].values())

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

			clean_words=lemmatizer.lemmatize(w,pos)            #[0]will give tuple #[1] will be noun
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
	
	print("here:",prediction)
	if(prediction=="pos"):
		resp = 'Positive'
	else:
		resp = 'Negative'
	return jsonify({'prediction': resp})

if __name__=='__main__':
	#app.run(host='192.168.0.192', port='5000', debug=True)
	app.run(host='0.0.0.0',)
