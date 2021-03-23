#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:17:13 2021

@author: andressa
"""

from keras.models import Model
from keras.layers import Dense, Input, Embedding, LSTM, Bidirectional, TimeDistributed, Dropout
import keras
import tensorflow.compat.v1 as tf
from transformers import TFBertModel
import numpy as np
from tagger_module import feature_to_index

def BertLayer(max_seq_length, model_name, number_layers=4, trainable=False, 
              from_pt=False, return_hidden_states=False):

    input_word_ids = Input(shape=(max_seq_length,), dtype=tf.int32,
                                         name="input_word_ids")
    input_mask = Input(shape=(max_seq_length,), dtype=tf.int32,
                                     name="input_mask")
    segment_ids = Input(shape=(max_seq_length,), dtype=tf.int32,
                                      name="segment_ids")
    
    bert_layer = TFBertModel.from_pretrained(model_name,from_pt=from_pt)
    
    if return_hidden_states:
      hidden_layers = bert_layer([input_word_ids, input_mask, segment_ids],
                                 output_hidden_states=return_hidden_states)[2]
      selected_layers = hidden_layers[-number_layers:]
      sequence_output = tf.keras.layers.Concatenate()(selected_layers)
    else:
      sequence_output,_ = bert_layer([input_word_ids, input_mask, segment_ids])
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=sequence_output)
    
    model.trainable = trainable
    
    return model

def generate_features_layers(features_names, input_len, hidden_dim,
                             scenario):
  layers = []
  features_map = feature_to_index(features_names, scenario)
  for feature in features_names:
    feature_name = feature + '_layer'
    output_len = len(features_map[feature])
    if feature == 'wordContextFeature':
      layer = OneHotModel(input_len, output_len, featureName=feature_name)
    else:
      layer = NumericFeatureLayer(input_len, output_len, hidden_dim, featureName=feature_name)
    layers.append(layer)
    
  return layers

def NumericFeatureLayer(maxLen, number_features, hidden_dim, featureName=None):

    initalization = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
    
    inputs = Input(shape=(maxLen,), dtype=tf.int32, name=featureName)
    embeddingLayer = Embedding(number_features, hidden_dim, input_length=maxLen,
                               embeddings_initializer=initalization)
    
    outputLayer = embeddingLayer(inputs)
    
    model = Model(inputs=inputs, outputs=outputLayer)
    
    return model

def OneHotModel(maxLen, number_features, featureName=None):
    inputs = Input(shape=(maxLen, number_features), dtype=np.int32, name=featureName)
    model = Model(inputs=inputs, outputs=inputs)
    
    return model

def BLSTM_Model(embedding_model, number_categories, features_models=[], 
                dropout=0.4, hidden_dim=128):

    if features_models == []:
      inputs = embedding_model.input
      embedding_layer = embedding_model.output
    else:
      models = [embedding_model] + features_models
      inputs = [model.input for model in models]
      embedding_layer = keras.layers.Concatenate(axis=-1, name='concat_layer')([model.output for model in models])
      
    blstm = Bidirectional(LSTM(hidden_dim, return_sequences=True))(embedding_layer)
    dropout_layer = Dropout(dropout)(blstm)
    outputs = TimeDistributed(Dense(number_categories, activation='softmax'))(dropout_layer)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    
    return model
  
  class parameters(object):
    
    def __init__(self, model_name, scenario, max_len=128, doc_stride=64, lstm_layer=512,
                 features_dim=75, number_bert_layers=4, num_epochs=50, batch=32,
                 learning_rate=1e-3):
        self.model = model_name
        self.scenario = scenario
        self.max_len = max_len
        self.doc_stride = doc_stride
        self.lstm_layer = lstm_layer
        self.features_dim = features_dim
        self.number_bert_layers = number_bert_layers
        self.num_epochs = num_epochs
        self.batch = batch
        self.learning_rate = learning_rate
        
    def __repr__(self):
        s = ("model name: {}\n"
             "scenario: {}\n"
             "max len: {}\n"
             "doc stride: {}\n"
             "lstm layer: {}\n"
             "features dim: {}\n"
             "number bert layers: {}\n"
             "number epochs: {}\n"
             "batch: {}\n"
             "learning rate: {}").format(self.model_name, self.scenario,
                                         self.max_len, self.doc_stride,
                                         self.lstm_layer, self.features_dim,
                                         self.number_bert_layers,self.num_epochs,
                                         self.batch,self.learning_rate)
        return s
