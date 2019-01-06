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
