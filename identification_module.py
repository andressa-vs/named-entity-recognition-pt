#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:55:12 2021

@author: andressa
"""

import re, nlpnet, pickle, nltk
import urllib.request, tarfile

local_file, html = urllib.request.urlretrieve('http://nilc.icmc.usp.br/nlpnet/data/pos-pt.tgz',
                                              filename='pos-tag')
tarfile.open(local_file).extractall("models/")

nltk.download('punkt')
nlpnet.set_data_dir('models/pos-pt/')

def mask_number(word):
  if word.isnumeric():
    return 'number'
  return word

def extract_word_features(sentence, index, pos_tag):
    special = '[$%&ºª€¥£₤]'
    current_word = sentence[index]
    prevWord = sentence[index-1].lower() if index-1 > 0 else 'BOS'
    nextWord = sentence[index+1].lower() if index+1 < len(sentence)-1 else 'EOS'
    prevTag = pos_tag[index-1] if index-1 > 0 else 'BOS'
    nextTag = pos_tag[index+1] if index+1 < len(sentence)-1 else 'EOS'
    features = {'isUpper' : current_word.isupper(),
                'isCap': current_word.istitle(),
            'isNumber': current_word.isnumeric(),
            'prevTag': prevTag,
            'posTag' : pos_tag[index],
            'nextTag': nextTag,
            'isLower': current_word.islower(),
            'isAlpha': current_word.isalpha(),
            'prevWord': mask_number(prevWord),
            'nextWord': mask_number(nextWord),
            'word': mask_number(current_word.lower()),
            'isMisc': True if not current_word.isupper() and not current_word.islower() and not current_word.istitle() else False,
            'isSpecial': True if re.search(special, current_word) else False}
    return features

def extract_identification_features(document):
    
    pos_tagging = pos_tagger.tag_tokens(document)
    
    assert len(pos_tagging) == len(document) 
      
    doc_features = [extract_word_features(document, j, pos_tagging) for j,_ in\
                    enumerate(document)]
        
    return doc_features

def find_entities(entities_prediction):
    
    entities_index = []
    i = 0
    
    while i < len(entities_prediction):
        if entities_prediction[i] == 'O':
            i += 1
        else:
            j = i
            
            while j < len(entities_prediction) and entities_prediction[j] == 'EN':
                j+= 1
                
            entities_index.append((i,j))
            i = j
            
    return  entities_index
    

def identify_named_entities(document):
    
    crfIdentifier = pickle.load(open('modelos/crfClassifier.pkl','rb'))
    
    identification_feats = extract_identification_features(document)
    predicted_entities = crfIdentifier.predict_single(identification_feats)
    
    assert len(document) == len(predicted_entities)
    
    entities_map = find_entities(predicted_entities)
    
    return entities_map


pos_tagger = nlpnet.POSTagger()
