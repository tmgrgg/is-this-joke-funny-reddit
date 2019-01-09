#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:53:44 2019

@author: griggles
"""

#Write a feature engineer that does TF_IDF... PCA... etc.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


class Fengineer:
    
    def __init__(self, tf_idf_rate=1, pca_num=None):
        self.tf_idf_rate = tf_idf_rate
        self.pca_num = pca_num
        
    def vectorize_jokes(self, jokes, index):
         vec = CountVectorizer()
         X = vec.fit_transform(jokes)
         return pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=index)
        
        
    def term_frequency(self, df):
        counts = df.sum(axis=0)
        count_dict = dict(counts)
        total_word_count = sum(count_dict.values())
    
        for ind, val in count_dict.items():
            count_dict[ind] = val/ total_word_count
    
        return count_dict
    
    def inverse_document_frequency(self, df):
        num_jokes = df.shape[0]

        idf_dict = {}
        num_docs_dict = {}
    
        for word in df:
            #calculate number of docs with t in it
            num_docs_containing = df[df[word] > 0].shape[0]
            idf_dict[word] = np.log(float(num_jokes)/float(num_docs_containing))
            num_docs_dict[word] = num_docs_containing
        
        return idf_dict, num_docs_dict

    def tf_idf(self, df):
        tf_idf_dict = {}
        tfs = self.term_frequency(df)
        idfs = self.inverse_document_frequency(df)[0]

        for word in df:
            tf_idf_dict[word] = tfs[word]*idfs[word]
    
        return tf_idf_dict
    
    def get_top_words_by_tf_idf(self, tf_idf_dict, n=2000):
        top_words = []
        ranked_words = list(map(lambda x: x[0], sorted(list(tf_idf_dict.items()), key=lambda x: x[1], reverse=True)))
    
        for i in range(n):
            top_words.append(ranked_words[i])
    
        return top_words
    
    def filter_by_tf_idf(self, df, n):
        top_words = self.get_top_words_by_tf_idf(self.tf_idf(df), n)
        return df[top_words]
    
    def perform_pca(self, df, n_components=None):
        if n_components:        
            pca = PCA(n_components)
            pca.fit(df)
            return pca.transform(df)
        else:
            return df
    
    def engineer_features(self, data):
        df_vec = self.vectorize_jokes(list(data['_joke']), data['_id'])
        
       # print('\n\VECT ::: VECT\n\n')
        
        #print(df_vec.head())
        #print(df_vec.shape)
        
        #print('\n\TF_IDF ::: TF_IDF\n\n')
        
        df_filtered = self.filter_by_tf_idf(df_vec, int(self.tf_idf_rate*df_vec.shape[1]))
        
        #print(df_filtered.head())
        #print(df_filtered.shape)
        
        #print('\n\PCA ::: PCA\n\n')
        
        array_pca = self.perform_pca(df_filtered, n_components=self.pca_num)
        
        df_pca = pd.DataFrame(array_pca, index=df_filtered.index)
        
       # print(df_pca.head())
       # print(df_pca.shape)
            
        df_labelled = pd.concat([df_pca, data['_label']], axis=1, join='inner')
        
        return df_labelled
      


      