from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import urllib
import numpy as np
import cv2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add
import ast

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
	
	#Model Architecture
	max_len=35
	vocab_size=1848
	in_text = "startseq"
	f=open('/home/chirag/Desktop/PROJECT/Image-Captioning/Flask_web_app/storage/word_to_index.txt')
	word_to_index=ast.literal_eval(f.read())
	f.close()
	f=open('/home/chirag/Desktop/PROJECT/Image-Captioning/Flask_web_app/storage/index_to_word.txt')
	index_to_word=ast.literal_eval(f.read())
	f.close()
	f = open("/home/chirag/Desktop/Machine_learning/DATASETS/glove6b50dtxt/glove.6B.50d.txt", encoding='utf8')
	embedding_index = {}

	print('---creating Embedding Index---')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype="float")
		
		embedding_index[word] = coefs
		
	f.close()
	print('---Loading Image Model---')
	Image_model = ResNet50(weights="imagenet", input_shape=(224,224,3))
	#print(Image_model.summary())
	model_new = Model(Image_model.input, Image_model.layers[-2].output)
	#defining encoding and preprocessing for image data
	def preprocess_image(img):
		img = image.load_img(img, target_size=(224,224))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)
		return img
	def encode_image(img):
		img = preprocess_image(img)
		feature_vector = model_new.predict(img)
		feature_vector = feature_vector.reshape(feature_vector.shape[1],)
		return feature_vector
	def get_embedding_output():
    
		emb_dim = 50
		embedding_output = np.zeros((vocab_size,emb_dim))
		
		for word, idx in word_to_index.items():
			embedding_vector = embedding_index.get(word)
			
			if embedding_vector is not None:
				embedding_output[idx] = embedding_vector
				
		return embedding_output
	
	
	embedding_output = get_embedding_output()
	input_img_fea = Input(shape=(2048,))
	inp_img1 = Dropout(0.3)(input_img_fea)
	inp_img2 = Dense(256, activation='relu')(inp_img1)
	input_cap = Input(shape=(max_len,))
	inp_cap1 = Embedding(input_dim=vocab_size, output_dim=50, mask_zero=True)(input_cap)
	inp_cap2 = Dropout(0.3)(inp_cap1)
	inp_cap3 = LSTM(256)(inp_cap2)
	decoder1 = add([inp_img2 , inp_cap3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)

	# Merge 2 networks
	model = Model(inputs=[input_img_fea, input_cap], outputs=outputs)
	model.layers[2].set_weights([embedding_output])
	model.layers[2].trainable = False
	model.compile(loss="categorical_crossentropy", optimizer="adam")
	model = load_model('/home/chirag/Desktop/PROJECT/Image-Captioning/Flask_web_app/model_weights/model_9.h5')
	#clf = joblib.load(NB_spam_model)
	if request.method == 'POST':
			
			message = request.form['message']
			data=(message)
			print(type(data))
			print(data)
			resp = urllib.request.urlopen('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtDxjmsiiWy3vwehEZmaO87eV5im3cWExevGPE0lr3cSbewFj5Vw')
			imag = np.asarray(bytearray(resp.read()), dtype="uint8")
			imag = cv2.imdecode(imag, cv2.COLOR_BGR2RGB)
			#imgi=encode_image(image)
			path_image='./Images/hey.png'
			cv2.imwrite(path_image,imag)
			f_v=encode_image(path_image)
			f_v=f_v.reshape((1,2048))

			for i in range(max_len):#max_len
				sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
				sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
				
				ypred =  model.predict([f_v,sequence])
				ypred = ypred.argmax()
				word = index_to_word[ypred]
				in_text+= ' ' +word
					
				if word =='endseq':
					break
					
					
			final_caption =  in_text.split()
			final_caption = final_caption[1:-1]
			final_caption = ' '.join(final_caption)
				
			print(final_caption)
	return render_template('result.html',prediction = final_caption)



if __name__ == '__main__':
	app.run(debug=True)