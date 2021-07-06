# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Optional
import string
string.punctuation

import os
from os.path import dirname, join, realpath
import joblib


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
import numpy as np
import pickle

nltk.download('stopwords')



# Declaring our FastAPI instance

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the given text input",
    version="0.1",
)


# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "models/sentiment_model.pkl"), "rb"
) as f:
    model = joblib.load(f)


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def preprocess_text(text, remove_stop_words=True, lemmatize_words=True):
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(" \d+", " ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace("\n", " ")
    text = "".join([i for i in text if i not in string.punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    #     Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        stem_text = [stemmer.stem(w) for w in text]
        text = " ".join(stem_text)

    return text

@app.post("/predict-sentiment")
def predict_sentiment(textinput: str):
    """
    A simple function that receive a content and predict the sentiment of the content.
    :param textinput:
    :return: prediction
    """

    # clean the review
    cleaned_review = preprocess_text(textinput)

    #perform tfidf
    tf1 = pickle.load(open("features/features.pkl", 'rb'))
    tf1_new = TfidfVectorizer(ngram_range=(1,2),max_features=10000,max_df=0.9,min_df=5, vocabulary = tf1.vocabulary_)
    X_tf1 = tf1_new.fit_transform([cleaned_review])

    # perform prediction
    prediction = model.predict(X_tf1)

    output = int(prediction)

    # output dictionary
    sentiments = {0: "Anger", 1: "Disgust",2: "Fear",3:"Guilt",4:"Joy",5:"Sadness",6:"Shame"}

    # show results
    result = {"prediction": sentiments[output]}


    return result





# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to Fusemachines!'}


# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name: str):
    return {'message': f'Welcome to Fusemachines!, {name}'}


@app.get('/model/{model_name}')
def read_model(model_name: str):
    return {"You are viewing": model_name}


class InputText(BaseModel):
    text: str


# POST METHODS
@app.post('/post-sentiment/')
def post_sentiment(text: InputText):
    return text
