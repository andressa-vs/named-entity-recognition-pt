#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:17:13 2021

@author: andressa
"""

from keras.models import Model
from keras.layers import Dense, Input, Embedding, LSTM, Bidirectional, TimeDistributed, Dropout
import keras
from transformers import TFBertModel
import numpy as np
from tensorflow.keras.optimizers import Adam, schedules
import tensorflow.compat.v1 as tf


class Blstm(object):
    
    def __init__(self, model_name, scenario, num_classes, max_len=128, doc_stride=64, lstm_layer=512,
                 features_dim=75, number_bert_layers=4, trainable_embedding=False, 
                 features=None, dropout=0.6):
        self.model_name = model_name
        self.scenario = scenario
        self.num_classes = num_classes
        self.max_len = max_len
        self.doc_stride = doc_stride
        self.lstm_layer = lstm_layer
        self.features_dim = features_dim
        self.number_bert_layers = number_bert_layers
        self.trainable_embedding = trainable_embedding
        self.features = features
        self.dropout = dropout
        self.from_pt = False
        self.output_hidden_states = False

        
        if self.model_name == 'neuralmind/bert-base-portuguese-cased':
            self.from_pt = True
        
        if self.number_bert_layers > 1:
            self.output_hidden_states = True
        
        self.generate_model()

        
    def __repr__(self):
        s = ("model name: {}\n"
             "scenario: {}\n"
             "max len: {}\n"
             "doc stride: {}\n"
             "lstm layer: {}\n"
             "features dim: {}\n"
             "number bert layers: {}\n"
             "trainable embedding: {}\n"
             "features names: {}\n"
             "num_classes: {}\n"
             "from pytorch: {}\n"
             "return hidden states: {}\n"
             "dropout: {}").format(self.model_name, self.scenario,
                                         self.max_len, self.doc_stride,
                                         self.lstm_layer, self.features_dim,
                                         self.number_bert_layers,
                                         self.trainable_embedding,
                                         self.features.keys(), self.num_classes,
                                         self.from_pt, self.return_hidden_states,
                                         self.dropout)
        return s
    
    def bert_embedding_layer(self):
        
        from_pt = self.from_pt
        output_hidden_states = self.output_hidden_states
    
        input_word_ids = Input(shape=(self.max_len,), dtype=tf.int32,
                                             name="input_word_ids")
        input_mask = Input(shape=(self.max_len,), dtype=tf.int32,
                                         name="input_mask")
        segment_ids = Input(shape=(self.max_len,), dtype=tf.int32,
                                          name="segment_ids")
        
        sequence_inputs = [input_word_ids, input_mask, segment_ids]
        bert_layer = TFBertModel.from_pretrained(self.model_name,from_pt=from_pt)
        bert_layer.config.output_hidden_states = output_hidden_states
        
        if output_hidden_states:
          hidden_layers = bert_layer(sequence_inputs)[2]
          selected_layers = hidden_layers[-self.number_bert_layers:]
          sequence_output = tf.keras.layers.Concatenate()(selected_layers)
        else:
          sequence_output,_ = bert_layer(sequence_inputs)
        
        bert_model = Model(inputs=sequence_inputs, outputs=sequence_output)
        
        bert_model.trainable = self.trainable_embedding
        
        return bert_model
    
    def numeric_feature_layer(self, number_features, featureName=None):
    
        initalization = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        
        inputs = Input(shape=(self.max_len,), dtype=tf.int32, name=featureName)
        embeddingLayer = Embedding(number_features, self.features_dim, input_length=self.max_len,
                                   embeddings_initializer=initalization)
        
        outputLayer = embeddingLayer(inputs)
        
        model = Model(inputs=inputs, outputs=outputLayer)
        
        return model   
    
    def one_hot_layer(self, number_features, featureName=None):
        inputs = Input(shape=(self.max_len, number_features), dtype=np.float32, name=featureName)
        model = Model(inputs=inputs, outputs=inputs)
        
        return model
    
    def features_layers(self): 
      layers = []
      
      for feature, dic in self.features.items():
        feature_name = feature + '_layer'
        output_len = len(dic)
        if feature == 'wordContextFeature':
          layer = self.one_hot_layer(output_len, featureName=feature_name)
        else:
          layer = self.numeric_feature_layer(output_len, featureName=feature_name)
        layers.append(layer)
        
      return layers
  
    def generate_model(self):
        
        self.features_models = self.features_layers()
        self.bert_model = self.bert_embedding_layer()
    
        if self.features_models == []:
          inputs = self.bert_model.input
          embedding_layer = self.bert_model.output
        else:
          models = [self.bert_model] + self.features_models
          inputs = [model.input for model in models]
          embedding_layer = keras.layers.Concatenate(axis=-1, name='concat_layer')([model.output for model in models])
          
        blstm = Bidirectional(LSTM(self.lstm_layer, return_sequences=True))(embedding_layer)
        dropout_layer = Dropout(self.dropout)(blstm)
        outputs = TimeDistributed(Dense(self.num_classes, activation='softmax'))(dropout_layer)
        
        self.blstm_model = Model(inputs=inputs, outputs=outputs)
        print(self.blstm_model.summary())
        

    def train(self, train_inputs, train_labels, validation_data=(), num_epochs=50, 
              learning_rate=1e-3, batch=32, decay_steps=1.0, decay_rate=1e-5):
        
        schedule = schedules.InverseTimeDecay(learning_rate, decay_steps, 
                                              decay_rate, staircase=False, 
                                              name=None)
        optimizer = Adam(learning_rate=schedule)
        
        self.blstm_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.blstm_model.fit(train_inputs, train_labels,
            validation_data= validation_data, epochs= num_epochs,
            batch_size=batch)

    def evaluate(self, evaluate_inputs, evaluate_labels, is_max_context, unwanted):
        
        unwanted_labels = [unwanted['X']] 
        
        logits = self.blstm_model.predict(evaluate_inputs, batch_size=1)
                    
        preds_mask = ((evaluate_labels != unwanted['X']) & (is_max_context))
        
        real_preds = [lab if lab not in unwanted_labels else unwanted['O'] for lab in np.argmax(logits[preds_mask], axis=-1)]
        real_labels = evaluate_labels[preds_mask]
          
        return real_labels, real_preds
