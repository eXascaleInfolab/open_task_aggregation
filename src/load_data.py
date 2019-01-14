#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 10:46:35 2018

@author: inesarous
"""

import nltk
import glob
import pandas as pd
import tweepy
import re
import os 
import errno
import numpy as np
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import iso639
import operator


class LoadData:
    def __init__(self, dataset):
        self.dataset = dataset
        self.answers=pd.read_csv(dataset)
        
        
    def authentification(self):
        consumer_key='O46hMtnXqTiVFNjeok87Tj78A'
        consumer_secret='4bJNSfMhlze9MlbBoXFxhXVKypGCO5piVmTpp1kqLsRdEfV3r8'
        access_token='1017774936017068033-fTasobi5ASvp9J5bwXZYFDl8BH9OC0'
        access_token_secret='S0BLFr00NccCE3baBXB2ToIH5S6EK7PAR7L5PSTE5ePEQ'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
        return api
    
    def text_process(self,raw_text,lang):
        try:
            """
            Takes in a string of text, then performs the following:
            1. Remove all punctuation
            2. Remove all stopwords
            3. Returns a list of the cleaned text
            """
            # Check characters to see if they are in punctuation
            nopunc = [char for char in list(raw_text) if char not in punctuation]
        
            # Join the characters again to form the string.
            nopunc = ''.join(nopunc)
            language=iso639.to_name(lang)
            languages = language.split(";", 1)
            # Now just remove any stopwords
            return [word for word in nopunc.lower().split() if word.lower() not in stopwords.words(languages[0].lower())]
        except ValueError as e:
            print (e)
    
    def remove_words(self,word_list):
        try:
            if (word_list!=None):
                remove = ['...','“','”','’','…']
                return [w for w in word_list if w not in remove]
            else:
                return ' '
        except ValueError as e:
            print (e)
    
    def word_count(self,sentence):
        if type(sentence)!=float:
            return len(sentence.split())
        
    # helper function to clean tweets
    def processTweet(self,tweet):
        # Remove HTML special entities (e.g. &amp;)
        tweet = re.sub(r'\&\w*;', '', tweet)
        #Convert @username to AT_USER
        tweet = re.sub('@[^\s]+','',tweet)
        # Remove tickers
        tweet = re.sub(r'\$\w*', '', tweet)
        # To lowercase
        tweet = tweet.lower()
        # Remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#\w*', '', tweet)
        # Remove Punctuation and split 's, 't, 've with a space for filter
        tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
        # Remove words with 2 or fewer letters
        tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
        # Remove whitespace (including new line characters)
        tweet = re.sub(r'\s\s+', ' ', tweet)
        # Remove single space remaining at the front of the tweet.
        tweet = tweet.lstrip(' ') 
        # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
        #tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
        return tweet
    
    
    def user_features_stats(self,api,user_pseudo,user_id):
        try:
            user = api.get_user(user_pseudo);
            user_acc_follower_nbr=user.followers_count;
            user_acc_followee_nbr=user.friends_count;
            user_acc_tweets_nbr=user.statuses_count;
            return user_acc_follower_nbr,user_acc_followee_nbr,user_acc_tweets_nbr
        except tweepy.TweepError as e:
            print("Error when sending tweet: %s" % e)   
            return 0,0,0
    
    def user_features_meta(self,api,user_pseudo,user_acc_tweets_nbr,user_id):     
        try:      
            columns = ['text','lang']
            bag= pd.DataFrame(columns=columns)
            all_tweets=api.user_timeline(screen_name = user_pseudo, count = 3200, include_rts = True)
            if (user_acc_tweets_nbr>0):
                for status in all_tweets:
                    bag=bag.append(pd.DataFrame([[status.text,status.lang]],columns=columns),ignore_index=True)
            counting_words= bag['text'].apply(self.word_count)       
            if (counting_words.empty==False):
                avg_length=counting_words.mean()            
                bag_txt=bag['text'].str.cat(sep=' ') 
                language=bag.lang.mode().iloc[0]
            else:
                avg_length=0
                bag_txt=''
                language='' 
            return bag_txt,avg_length,language
        except tweepy.TweepError as e:
            print("Error when sending tweet: %s" % e)
            user_id=user_id+1
            return ' ',0,'null'
    
    def bag_words_tweets(self,tweets):
        tweet_text_processed=pd.DataFrame(index=range(0,tweets.shape[0]),columns=['text','tokens'])
        for id_tweet in range(0,tweets.shape[0]):
            if (tweets['text'][id_tweet]!=' '):
                tweet_text_processed['text'].iloc[id_tweet]=self.processTweet(tweets['text'][id_tweet])
                tweet_text_processed['tokens'].iloc[id_tweet] =self.text_process(tweet_text_processed['text'].iloc[id_tweet],tweets['lang'][id_tweet])
                tweet_text_processed['tokens'].iloc[id_tweet]=self.remove_words(tweet_text_processed['tokens'].iloc[id_tweet])
            else:
                tweet_text_processed['text'].iloc[id_tweet]=' '
                tweet_text_processed['tokens'].iloc[id_tweet]=' '       
        # vetorize
        bow_transformer = CountVectorizer().fit(tweet_text_processed['text'])
        # transform the entire DataFrame of messages
        messages_bow = bow_transformer.transform(tweet_text_processed['text'])
        tfidf_transformer = TfidfTransformer().fit(messages_bow)
        # to transform the entire bag-of-words corpus
        bag_words = tfidf_transformer.transform(messages_bow)
        return bag_words,bow_transformer.vocabulary_

    def generate_annotation_matrix(self,dataset):
        answers=pd.read_csv(dataset);
        true_labels=answers[['infl_acc','label']].drop_duplicates(subset=['infl_acc'])
        all_workers=answers.user_acc.unique();
        all_infl=answers.infl_acc.unique();
        aij=np.zeros((all_workers.shape[0]*all_infl.shape[0],3))
        aij[:, 0] = np.repeat(np.array(range(0, all_workers.shape[0])), all_infl.shape[0])
        aij[:, 1] = np.tile(np.array(range(0, all_infl.shape[0])), all_workers.shape[0])
        for worker in range(0, all_workers.shape[0]):
            worker_acc = all_workers[worker]
            named_infl = answers[answers['user_acc'] == worker_acc]['infl_acc'].iloc[0]
            index = np.where(all_infl == named_infl)
            aij[((worker * all_infl.shape[0]) + index[0][0]), 2] = 1
        return aij,all_workers,all_infl,true_labels.label

    #input to the model
    def run_load_data(self):
        api=self.authentification()
        columns = ['text','lang']
        all_workers_tweets= pd.DataFrame(columns=columns)
        options = {'never':0,
                   'rarely':20,
                   'sometimes':50,
                   'always':80,
                }
        worker_x=pd.DataFrame(columns=['follower_nbr','followee_nbr','tweets_nbr',\
                                      'avg_length_tweets','language','exp_coeff','conn_coeff','mot_coeff'])
            
        for worker_id in range (0,self.answers.shape[0]):
            worker_pseudo=self.answers['user_acc'][worker_id]
            
            #social media features
            [worker_acc_follower_nbr,worker_acc_followee_nbr,worker_acc_tweets_nbr]=self.user_features_stats(api,worker_pseudo,worker_id)
            [bag_txt,avg_length,language]=self.user_features_meta(api,worker_pseudo,worker_acc_tweets_nbr,worker_id);
            
            #coefficients from the survey
            exp_coeff=self.answers['exp'][worker_id]
            conn_coeff=(options[self.answers['freq'][worker_id]]+self.answers['conn'][worker_id])/2;
            mot_coeff=self.answers['mot'][worker_id]
            
            all_workers_tweets=all_workers_tweets.append(pd.DataFrame([[bag_txt,language]],columns=columns),ignore_index=True,sort=True)
            
            #worker features
            worker_x=worker_x.append(pd.DataFrame([[worker_pseudo,worker_acc_follower_nbr,worker_acc_followee_nbr,worker_acc_tweets_nbr,avg_length,\
                                                    language,exp_coeff,conn_coeff,mot_coeff]],columns=['user_name','follower_nbr',\
                                                    'followee_nbr','tweets_nbr','avg_length_tweets','language','exp_coeff','conn_coeff',\
                                                    'mot_coeff']),ignore_index=True,sort=True)
        [bag_words_worker,vocab]= self.bag_words_tweets(all_workers_tweets)
        sorted_vocab_worker=sorted(vocab.items(), key=operator.itemgetter(1),reverse=True)
        worker_x = pd.concat([worker_x, pd.DataFrame(bag_words_worker.toarray(),columns=[idx for idx, val in sorted_vocab_worker])], axis=1)
        worker_x = pd.concat([worker_x, pd.DataFrame(bag_words_worker.toarray(),columns=[idx for idx, val in sorted_vocab_worker])], axis=1)

        
        infl_x=pd.DataFrame(columns=['follower_nbr','followee_nbr','tweets_nbr',\
                                      'avg_length_tweets','language'])
        columns = ['text','lang']
        all_infl_tweets= pd.DataFrame(columns=columns)
        for inlfuencer_id in range (0,self.answers.shape[0]):
            influencer_pseudo=self.answers['infl_acc'][inlfuencer_id]
            if (influencer_pseudo!=''):
                [infl_acc_follower_nbr,infl_acc_followee_nbr,infl_acc_tweets_nbr]=self.user_features_stats(api,influencer_pseudo,inlfuencer_id);
                print (influencer_pseudo,infl_acc_follower_nbr,infl_acc_followee_nbr)
                [bag_txt,avg_length,language]=self.user_features_meta(api,influencer_pseudo,infl_acc_tweets_nbr,inlfuencer_id);
                all_infl_tweets=all_infl_tweets.append(pd.DataFrame([[bag_txt,language]],columns=columns),ignore_index=True,sort=True)
                infl_x=infl_x.append(pd.DataFrame([[influencer_pseudo,infl_acc_follower_nbr,infl_acc_followee_nbr,infl_acc_tweets_nbr,avg_length,\
                                                    language]],columns=['user_name','follower_nbr',\
                                                    'followee_nbr','tweets_nbr','avg_length_tweets','language']),ignore_index=True,sort=True)
        
        [bag_words_infl,vocab]= self.bag_words_tweets(all_infl_tweets)
        sorted_vocab_infl=sorted(vocab.items(), key=operator.itemgetter(1),reverse=True)    
        infl_x = pd.concat([infl_x, pd.DataFrame(bag_words_infl.toarray(),columns=[idx for idx, val in sorted_vocab_infl])], axis=1)
        worker_x=worker_x.drop(['language'], axis=1)
        print(infl_x[['follower_nbr', 'followee_nbr']])
        infl_x = infl_x.drop_duplicates(subset=['user_name'])
        infl_x=infl_x.drop(['language','user_name'], axis=1)
        return worker_x,infl_x
        #print bag_txt,worker_id
        #tweets=tweets.append(pd.DataFrame([[bag_txt,language]],columns=columns),ignore_index=True)
        #train_data[i,:]=[user_acc_follower_nbr,user_acc_followee_nbr,user_acc_tweets_nbr,avg_length]

