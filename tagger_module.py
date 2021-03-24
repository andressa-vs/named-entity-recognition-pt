#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:38:05 2021

@author: andressa
"""

from collections import defaultdict
from identification_module import identify_named_entities, pos_tagger
from rule_based_module import classify_contexts


POS_TAGS = ['ADJ','ART','ADV-KS','IN','KS','NPROP','PCP','PREP',
             'PREP+ART','PREP+PRO-KS','PREP+PROPESS',
             'PROADJ','PRO-KS-REL','PROSUB','V','ADV',
             'ADV-KS-REL','CUR','KC','N','NUM','PDEN',
             'PREP+ADV','PREP+PROADJ','PREP+PRO-KS-REL',
             'PREP+PROSUB','PRO-KS','PROPESS','PU','VAUX']

WORD_TYPE_TAGS = ['ALPHA','NUMERIC','ALPHA-NUM','NON-ALPHA']

WORD_CASE_TAGS = ['UPPER','FIRST-UPPER','LOWER', 'MISC']

TOTAL_CLASSES = ['PESSOA','LOCAL','ORGANIZAÇÃO', 'TEMPO','VALOR','OUTRO',
           'COISA','ABSTRAÇÃO', 'ACONTECIMENTO']

SELECTIVE_CLASSES = ['PESSOA','LOCAL','ORGANIZAÇÃO','TEMPO','VALOR']

def tag_encoder(scenario):
    
    if scenario.lower() not in ["selective","total"]:
        raise ValueError("Scenario must be 'selective' or 'total'")
    
    entities_tags = ['X', 'O']
    
    if scenario == 'selective':
        classes = SELECTIVE_CLASSES
    else:
        classes = TOTAL_CLASSES
    
    for entity_class in classes:
        begin_class = 'B-{}'.format(entity_class[:3])
        end_class = 'I-{}'.format(entity_class[:3])
        entities_tags.extend((begin_class, end_class))
        
    return entities_tags

def get_word_type(token):
    
    if token.isalpha():
      wrd_type = 'ALPHA'
    elif token.isnumeric():
      wrd_type = 'NUMERIC'
    elif token.isalnum():
      wrd_type = 'ALPHA-NUM'
    else:
      wrd_type = 'NON-ALPHA'
       
    return wrd_type

def get_word_case(token):
    
  if token.isupper():
    wrd_case = 'UPPER'
  elif token.istitle():
    wrd_case = 'FIRST-UPPER'
  elif token.islower():
    wrd_case = 'LOWER'
  else:
    wrd_case = 'MISC'
    
  return wrd_case


class features(object):
    
    def __init__(self, features, scenario):
        self.features = features
        self.scenario = scenario
        self.get_features()
        self.feature_to_index()
        self.features_names = self.features_dict.keys()
    
    def get_features(self):
        self.features_dict = {}
        
        if "ortographic_features" in self.features:
            self.features_dict["wordTypeFeature"] = ['X'] + WORD_TYPE_TAGS
            self.features_dict["wordCaseFeature"] = ['X'] + WORD_CASE_TAGS
        if "pos_tag_feature" in self.features:
            self.features_dict["posTagFeature"] = ['X'] + POS_TAGS
        if "word_context_feature" in self.features:
            out_feats = ['X', 'O']
            if self.scenario == 'total':
                self.features_dict["wordContextFeature"] = out_feats + TOTAL_CLASSES
            elif self.scenario == 'selective':
                self.features_dict["wordContextFeature"] = out_feats + SELECTIVE_CLASSES
                
    def feature_to_index(self):
        self.map_features = {}
        
        for feature, value in self.features_dict.items():
            self.map_features[feature] = {label:index for index,label in enumerate(value)}
            
        

def extract_extra_features(doc_tokens, features_names, scenario):
    
    features = defaultdict(list)
    
    if 'ortographic_features' in features_names:        
        for token in doc_tokens:
            features['wordTypeFeature'] += [get_word_type(token)]
            features['wordCaseFeature'] += [get_word_case(token)]
            
    if 'pos_tag_feature' in features_names:
        features['posTagFeature'] = pos_tagger.tag_tokens(doc_tokens)

    
    if 'word_context_feature' in features_names:
        doc_entities = identify_named_entities(doc_tokens)
        features['wordContextFeature'] = classify_contexts(doc_tokens,
                                                             doc_entities,
                                                             scenario)   
    return features
        