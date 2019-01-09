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
    
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
english_words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

#return cleaned tokens
def clean_joke(joke, english):
   raw_tokens = nltk.regexp_tokenize(joke, pattern)
   tokens = [t.lower() for t in raw_tokens]
   listed = [t for t in tokens if not t in stop_words]
   if english:
       listed = list(filter(lambda w: w in english_words, listed))
   return listed

#return stemmed tokens
stemmer = SnowballStemmer("english")
def stem_joke(joke, english):
    listed = clean_joke(joke, english)
    stemmed = [stemmer.stem(word) for word in listed]
    return stemmed

#return lemmed tokens
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize_joke(joke, english):
    listed = clean_joke(joke, english)
    lemmatized = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in listed]
    return lemmatized

def untokenize(tokens):
    return ' '.join(tokens)

#stem = none, lem or stem depending on the type of stemmization we want to do on the words
#english boolean-value telling us whether we want to filter for english words prior to stemmization
#rough values: 35,000 unq for ('lem', True)-  16,000 for ('lem', False)
    #29,000 for ('stem', True) - #9,000 got ('stem', False)
def transform_data(df, stem='stem', english=True):
    if (stem == 'lem'):
        df['joke'] = df.apply(lambda row: untokenize(lemmatize_joke(row['title'] + ' ' + row['body'], english)), axis=1)
    elif (stem == 'none'):
        df['joke'] = df.apply(lambda row: untokenize(clean_joke(row['title'] + ' ' + row['body'], english)), axis=1)
    else:
        df['joke'] = df.apply(lambda row: untokenize(stem_joke(row['title'] + ' ' + row['body'], english)), axis=1)

    #add underscore to column names
    df.index = df['id']
    df.columns = ['_' + x for x in list(df)]
    return df