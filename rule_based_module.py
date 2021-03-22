
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:57:50 2021

@author: andressa
"""

from listas import get_classifiers


AGNOMES = ('jr','junior','primeiro','segundo','I','II',
                        'filho','neto')

EMPRESAS = ('ltda','sa','me','mei','epp','eireli')


def ort_lex_features(ne):
    last_en_token = ne[-1]
    
    if "&" in ne:
        return 'organização'
    if last_en_token in EMPRESAS:
        return 'organização'
    if last_en_token in AGNOMES:
        return 'pessoa'

    return 'O'

def match_contexts(first_index, last_index, sentence, scenario):
        
    first_en_token = sentence[first_index].lower() 
    prevWord = sentence[first_index-1].lower() if first_index-1 > 0 else 'NONE'
    ne = ' '.join(sentence[first_index:last_index])
    
    if scenario == 'total':
        classifiers = get_classifiers()
    elif scenario == 'selective':
        classifiers = get_classifiers(lists_names=['parentesco','pronome','profissão',
                                       'estabelecimento', 'geomorfologia',
                                       'logradouro'])
        
    for cat, dic  in classifiers.items():
        clas, subclas = cat.split('|')
        if prevWord in dic['singular'] or first_en_token in dic['singular']:
            return clas, 'sg'
        elif prevWord in dic['plural'] or first_en_token in dic['plural']:
            return clas, 'pl'
    
    ort_lex_feat = ort_lex_features(ne)
    
    if ort_lex_feat != 'O':
        return ort_lex_feat, 'ort'
    
    return 'O', None
                

    
def is_sequence(document, prev_index, doc_entities):
    sequence = 0
    e_bool = False
    
    for first_index, last_index in doc_entities:
        if first_index == prev_index+1:
            if document[prev_index] == 'e':
                e_bool = True
                sequence += 1
                break    
            elif document[prev_index] == ',':
                sequence += 1
        else:
            break
        prev_index = last_index
    
    if e_bool:
        return sequence
    
    return 0
        
    
def classify_contexts(document, doc_entities, scenario):
    
    context_features = ['O'] * len(document)
       
    sequence = 0
    clas = 'O'
    
    for i, (first_index, last_index) in enumerate(doc_entities):
        if sequence == 0:
            clas, ft = match_contexts(first_index,last_index,document,scenario)
        
        if clas != 'O':
            ne_len = last_index - first_index
            context_features[first_index:last_index] = [clas.upper()] * ne_len   
            
        if ft == 'pl':
            sequence = is_sequence(document,last_index,doc_entities[i+1:])
            
    return context_features

        