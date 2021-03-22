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

def convert_to_ids(items, start=0):
    return {k:i for i, k in enumerate(items, start=start)}

def tag_encoder(scenario):
    
    if scenario.lower() not in ["selective","total"]:
        raise ValueError("Scenario must be selective or total")
    
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


def feature_to_index(features_names, scenario):
    dict_map_feature = {}

    if 'ortographic_features' in features_names:
        wordType = ['X'] + WORD_TYPE_TAGS
        wordCase = ['X'] + WORD_CASE_TAGS
        dict_map_feature['wordTypeFeature'] = convert_to_ids(wordType)
        dict_map_feature['wordCaseFeature'] = convert_to_ids(wordCase)
    if 'pos_tag_feature' in features_names:
        posTag = ['X'] + POS_TAGS       
        dict_map_feature['posTagFeature'] = convert_to_ids(posTag)
    if 'word_context_feature' in features_names:
        if scenario == 'total':
            classes = ['X', 'O'] + TOTAL_CLASSES
        elif scenario == 'selective':
            classes = ['X', 'O'] + SELECTIVE_CLASSES
        else:
            classes = []
        
        dict_map_feature['wordContextFeature'] = convert_to_ids(classes)
    return dict_map_feature
        