#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:09:13 2021

@author: andressa
"""

from transformers import BertTokenizer
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from models import BertLayer, generate_features_layers 
from pre_processing import convert_example_to_spans
import pandas as pd
from collections import defaultdict
  
def get_training_samples(df, doc_type):
    
    agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                          s["Tag"].values.tolist())]
    grouped = df.groupby("{} #".format(doc_type),sort=False).apply(agg_func)
    
    examples = []
    
    for doc in grouped:
        doc_tokens = [w for w,_ in doc]
        doc_labels = [l for _,l in doc]
        examples.append((doc_tokens, doc_labels))
        
    return examples

def split_features_extracted(features_dict, random_state):
  train_features, valid_features = [], []
  for name, feats in features_dict.items():
    dtype = np.int32
    if name == 'wordContextFeature':
      feats = to_categorical(feats, dtype=np.float32)
      dtype = np.float32
    train, valid = create_dataset(feats, random_state=random_state,dtype=dtype)
    train_features.append(train)
    valid_features.append(valid)
    
  return train_features, valid_features

def create_dataset(list_feats, random_state=2020, test_size=0.1,dtype=np.int32):

  train, test = train_test_split(list_feats, random_state=random_state, 
                                 test_size=test_size)
  return np.asarray(train, dtype=dtype), np.asarray(test, dtype=dtype)


def features_names(features):
    features_names = []
    
    for feature in features:
        if feature == 'ortographic_features':
            features_names.extend(['wordTypeFeature','wordCaseFeature'])
        elif feature == 'pos_tag_feature':
            features_names.append('posTagFeature')
        elif feature == 'word_context_feature':
            features_names.append('wordContextFeature')
    
    return features_names

def get_train_inputs(all_features, extra_features, training=True):
    
    features_dic = defaultdict(list)
    
    for doc_feature in all_features:
        features_dic['inputs_ids'].append(doc_feature['input_ids'])
        features_dic['mask_ids'].append(doc_feature['masks'])
        features_dic['segment_ids'].append(doc_feature['segment_ids'])
        features_dic['labels_ids'].append(doc_feature['label_ids'])
        for feature in extra_features:
            features_dic[feature].append(doc_feature[feature])
            
        if not training:
            features_dic['max_context'].append(doc_feature['token_is_max_context'])
    
    return features_dic
        
        
#MODEL_NAME = 'bert-base-multilingual-cased'
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
MAX_LEN = 128
DOC_STRIDE = 64
SCENARIO = 'selective'
CORPUS_PATH = "data/first-harem-filtered.csv"
TEXT_TYPE = 'Doc'
TRAINING = True

EXTRA_FEATURES = ['pos_tag_feature',
                  'ortographic_features',
                  'word_context_feature']

dataset = pd.read_csv(CORPUS_PATH,sep='\t').dropna()
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

extra_features = EXTRA_FEATURES
extra_features_names = features_names(extra_features)

train_examples = get_training_samples(dataset, TEXT_TYPE)[:5]
all_train_features = convert_example_to_spans(train_examples,tokenizer,SCENARIO,
                                          extra_features, doc_max_len=MAX_LEN,
                                          doc_stride=DOC_STRIDE)

train_features = get_train_inputs(all_train_features, extra_features_names,
                                TRAINING)

train_inputs, valid_inputs = create_dataset(train_features['inputs_ids'])
train_labels, valid_labels = create_dataset(train_features['labels_ids'])
train_segments, valid_segments = create_dataset(train_features['segment_ids'])
train_masks, valid_masks = create_dataset(train_features['mask_ids'])







