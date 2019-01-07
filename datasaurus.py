#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:42:30 2019

@author: griggles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re

def plot_dist(drop_threshold=0):
    df = data
    df = df[df['score'] >= drop_threshold]
    df['score'].hist(bins=20)

def sample_data(funny_threshold=50):
    df = pd.read_json('data/reddit_jokes.json')
    funnies = df[df['score'] >= funny_threshold]
    funnies['label'] = 1
    duds = df[df['score'] < funny_threshold]
    duds['label'] = 0
    
    #downsample minority class
    if (funnies.shape[0] < duds.shape[0]):
        duds = duds.sample(funnies.shape[0])
    else:
        funnies = funnies.sample(duds.shape[0])
    
    return funnies.append(duds)
    
wordnet_lemmatizer = WordNetLemmatizer()
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
english = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

  
def lemmatize_joke(joke):
   raw_tokens = nltk.regexp_tokenize(joke, pattern)
   tokens = [t.lower() for t in raw_tokens]
   listed = [t for t in tokens if not t in stop_words]
   lemmatized = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in listed]
   #words = list(filter(lambda w: w in english, lemmatized))
   return " ".join(lemmatized)

def transform_data(df):
    df['joke'] = df.apply(lambda row: lemmatize_joke(row['title'] + ' ' + row['body']), axis=1)