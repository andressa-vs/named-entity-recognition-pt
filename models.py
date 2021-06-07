import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Bidirectional, TimeDistributed, Concatenate
from tensorflow.keras.initializers import RandomUniform
from transformers import TFBertModel
import numpy as np
from tensorflow.keras.optimizers import Adam, schedules

from tf2crf import CRF, ModelWithCRFLoss


class BlstmForNer(object):
    
    def __init__(self, bert_model_path, labels, 
                 max_len=128, 
                 lstm_layer=512,
                 features_dim=75, 
                 number_bert_layers=4, 
                 bert_is_trainable=False, 
                 features=None, 
                 dropout=0.6):
        
        self.bert_model_path = bert_model_path
        self.labels = labels
        self.max_len = max_len
        self.lstm_layer = lstm_layer
        self.features_dim = features_dim
        self.number_bert_layers = number_bert_layers
        self.bert_is_trainable = bert_is_trainable
        self.features = features
        self.dropout = dropout
        self.from_pt = False
        self.output_hidden_states = False
        
        bert_portuguese_cased = 'neuralmind/bert-base-portuguese-cased'
        
        if self.bert_model_path == bert_portuguese_cased:
            self.from_pt = True
        
        if self.number_bert_layers > 1:
            self.output_hidden_states = True
        
        self.lab_to_ind = self.labels_to_index()
        self.ind_to_lab = self.index_to_labels()
        
        
    def __repr__(self):
        s = ("model name: {}\n"
             "max len: {}\n"
             "lstm layer: {}\n"
             "features dim: {}\n"
             "number bert layers: {}\n"
             "trainable embedding: {}\n"
             "features names: {}\n"
             "labels: {}\n"
             "from pytorch: {}\n"
             "return hidden states: {}\n"
             "dropout: {}").format(self.bert_model_path, 
                                   self.max_len, 
                                    self.lstm_layer, self.features_dim,
                                    self.number_bert_layers,
                                    self.bert_is_trainable,
                                    self.features.keys(), self.labels,
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
        bert_layer = TFBertModel.from_pretrained(self.bert_model_path,from_pt=from_pt,
                                                 output_hidden_states=self.output_hidden_states)
        
        if output_hidden_states:
          hidden_layers = bert_layer(sequence_inputs)[2]
          selected_layers = hidden_layers[-self.number_bert_layers:]
          sequence_output = Concatenate()(selected_layers)
        else:
          sequence_output,_ = bert_layer(sequence_inputs)
        
        bert_model = Model(inputs=sequence_inputs, outputs=sequence_output)
        
        bert_model.trainable = self.bert_is_trainable
        
        return bert_model
    
    def numeric_feature_layer(self, number_features, featureName=None):
    
        initalization = RandomUniform(minval=-0.5, maxval=0.5)
        
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
  
    def compute_mask(self, inputs):
        
        mask_layer = Embedding(30, 10, input_length=self.max_len, mask_zero=True)
        
        if len(inputs) == 3:
            mask = mask_layer.compute_mask(inputs[1])
        else:
            mask = mask_layer.compute_mask(inputs[3])
            
        return mask
  
    def _build_model(self):
       
        self.bert_model = self.bert_embedding_layer()
        self.features_models = self.features_layers()
        
        if self.features_models == []:
          inputs = self.bert_model.input
          embedding_layer = self.bert_model.output
        else:
          models = [self.bert_model] + self.features_models
          inputs = [model.input for model in models]
          embedding_layer = Concatenate(axis=-1, name='concat_layer')([model.output for model in models])
         
        mask = self.compute_mask(inputs) 
          
        blstm = Bidirectional(LSTM(self.lstm_layer, return_sequences=True, dropout=self.dropout))(embedding_layer, mask=mask)
        outputs = TimeDistributed(Dense(len(self.labels), activation='softmax'))(blstm)
        
        self.blstm_model = Model(inputs=inputs, outputs=outputs)
        print(self.blstm_model.summary())
        

    def train(self, train_inputs, train_labels, validation_data=(), 
              num_epochs=50, learning_rate=1e-3, batch=32, 
              decay_steps=1.0, decay_rate=1e-5,callback=[]):
        
        schedule = schedules.InverseTimeDecay(learning_rate, 
                                              decay_steps, 
                                              decay_rate, 
                                              staircase=False, 
                                              name=None)
        
        optimizer = Adam(learning_rate=schedule)
        
        self.blstm_model.compile(optimizer=optimizer, 
                                 loss='categorical_crossentropy', 
                                 metrics=['accuracy'])

        self.blstm_model.fit(train_inputs, 
                             train_labels,
                             validation_data=validation_data, 
                             epochs= num_epochs,
                             batch_size=batch,
                             callbacks=callback)
        
    def labels_to_index(self):
        return {label: index for index, label in enumerate(self.labels)}
    
    def index_to_labels(self):
        return {index: label for index, label in enumerate(self.labels)}
    
    def convert_to_labels(self, ids_list):
        return [self.ind_to_lab[idx] for idx in ids_list]
    
    def convert_to_ids(self, labels_list):
        return [self.lab_to_ind[label] for label in labels_list]
        
    def evaluate(self, evaluate_inputs, evaluate_labels, is_max_context, batch_size=32):
    
        x_label = self.lab_to_ind['X']
        o_label = self.lab_to_ind['O']
        
        logits = self.blstm_model.predict(evaluate_inputs, batch_size=batch_size)
                    
        preds_mask = ((evaluate_labels != [x_label]) & (is_max_context))
        
        real_preds_idx = [lab if lab != x_label else o_label for lab in np.argmax(logits[preds_mask], axis=-1)]
        real_labels_idx = evaluate_labels[preds_mask]
        
        assert len(real_preds_idx) == len(real_labels_idx)
        
        real_preds_tags = self.convert_to_labels(real_preds_idx)
        real_labels_tags = self.convert_to_labels(real_labels_idx)
          
        return real_preds_tags, real_labels_tags

class BlstmForNerCRF(BlstmForNer):
    
    def __init__(self, bert_model_path, labels,**kwargs):
        super().__init__(bert_model_path, labels, **kwargs)
        self.crf = CRF(len(self.labels))
            
    def _build_model(self):
         
        self.bert_model = self.bert_embedding_layer()
        self.features_models = self.features_layers()
        
        if self.features_models == []:
          inputs = self.bert_model.input 
          embedding_layer = self.bert_model.output
          
        else:
          models = [self.bert_model] + self.features_models
          inputs = [model.input for model in models]
          embedding_layer = Concatenate(axis=-1, name='concat_layer')([model.output for model in models])
         
        mask = self.compute_mask(inputs)
          
        blstm = Bidirectional(LSTM(self.lstm_layer, return_sequences=True, dropout=self.dropout))(embedding_layer, mask=mask)
        time_dist = TimeDistributed(Dense(self.lstm_layer, activation=None))(blstm)
        outputs = self.crf(time_dist)
        
        blstm_model = Model(inputs=inputs, outputs=outputs)
        self.blstm_crf_model = ModelWithCRFLoss(blstm_model, sparse_target=True)

        print(blstm_model.summary())
        
    def train(self, train_inputs, train_labels, validation_data=(), num_epochs=50, 
              learning_rate=1e-3, batch=32, decay_steps=1.0, decay_rate=1e-5,
              callback=[]):
        
        schedule = schedules.InverseTimeDecay(learning_rate, 
                                              decay_steps, 
                                              decay_rate, 
                                              staircase=False)
        optimizer = Adam(learning_rate=schedule)
        
        self.blstm_crf_model.compile(optimizer=optimizer)

        self.blstm_crf_model.fit(train_inputs, train_labels,
            validation_data= validation_data, epochs= num_epochs,
            batch_size=batch, callbacks=callback)

    def evaluate(self, evaluate_inputs, evaluate_labels, is_max_context, batch_size=32):
    
        x_label = self.lab_to_ind['X']
        o_label = self.lab_to_ind['O']
        
        logits = self.blstm_crf_model.predict(evaluate_inputs, batch_size=batch_size)
                    
        preds_mask = ((evaluate_labels != [x_label]) & (is_max_context))
        
        real_preds_idx = [lab if lab != x_label else o_label for lab in logits[preds_mask]]
        real_labels_idx = evaluate_labels[preds_mask]
        
        assert len(real_preds_idx) == len(real_labels_idx)
        
        real_preds_tags = self.convert_to_labels(real_preds_idx)
        real_labels_tags = self.convert_to_labels(real_labels_idx)
          
        return real_preds_tags, real_labels_tags
