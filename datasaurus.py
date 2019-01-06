#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:42:30 2019

@author: griggles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_json('data/reddit_jokes.json')

def plot_dist(drop_threshold=0):
    df = data
    df = df[df['score'] >= drop_threshold]
    df['score'].hist(bins=20)

def build_data(funny_threshold=50):
    df = data
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