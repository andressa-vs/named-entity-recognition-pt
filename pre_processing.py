#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:43:07 2021

@author: andressa
"""


from collections import namedtuple, defaultdict
from tagger_module import tag_encoder, extract_extra_features, feature_to_index



#Código extraído de https://github.com/anupamsingh610/bert_ner_stride/blob/4865c0229c344ee4e1c70cf03dd1d248ebdd49f9/BERT_NER_STRIDE.py#L249

def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def insert_X_features(extra_features, all_prediction_mask):
    all_extra_features = defaultdict(list)
    for feature, values in extra_features.items():
        i = 0
        for prediction in all_prediction_mask:
            if prediction == True:
                all_extra_features[feature].append(values[i])
                i += 1
            else:
                all_extra_features[feature].append('X')
    return all_extra_features


def convert_example_to_spans(examples, 
                           tokenizer, 
                           scenario,
                           extra_features_names=[], 
                           doc_max_len=512, 
                          doc_stride=128):
    

    all_docs_features = []         
    tags_list = tag_encoder(scenario)
    tags_to_ids = {index: tag for tag, index in enumerate(tags_list)}
    map_features = feature_to_index(extra_features_names, scenario)   
    
    
    for doc_index, (doc_tokens, doc_labels) in enumerate(examples):
        
        print("Preprocessing document %s"%(doc_index+1))
        
        all_doc_tokens = []
        all_doc_labels = []
        all_prediction_mask = []
    
        extra_features = extract_extra_features(doc_tokens, extra_features_names,
                                                scenario)
        
        for i, token in enumerate(doc_tokens):
            
            subtokens = tokenizer.tokenize(token)
            
            for j, subtoken in enumerate(subtokens):
                
                all_doc_tokens.append(subtoken)
                all_prediction_mask.append(j== 0)
                
                if j == 0:
                    all_doc_labels.append(doc_labels[i])
                    
                else:
                    all_doc_labels.append('X')
        
                    
        all_extra_features = insert_X_features(extra_features, 
                                               all_prediction_mask)
        
                    
        assert len(all_doc_tokens) == len(all_prediction_mask)
        assert len(all_doc_tokens) == len(all_doc_labels)
        
        if extra_features_names != []:
            for doc_features in all_extra_features.values():
                assert len(all_doc_tokens) == len(doc_features)
                        
                
        max_tokens_for_doc = doc_max_len - 1
      
        _DocSpan = namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        
        while start_offset < len(all_doc_tokens):
          length = len(all_doc_tokens) - start_offset
          if length > max_tokens_for_doc:
            length = max_tokens_for_doc
          doc_spans.append(_DocSpan(start=start_offset,length=length))
          if start_offset + length == len(all_doc_tokens):
            break
          start_offset += min(length, doc_stride)
        
        
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            labels_ids = []
            segment_ids = []
            token_is_max_context = []
            extra_features_dic = {}
            tokens.append('[CLS]')
            labels_ids.append(tags_to_ids['X'])
            segment_ids.append(0)
            token_is_max_context.append(False)
          
    
            for feature in all_extra_features:
                extra_features_dic[feature] = [0]
                  
            for i in range(doc_span.length):
                split_token_idx = doc_span.start + i
                is_max_context = check_is_max_context(doc_spans, doc_span_index,split_token_idx)
                token_is_max_context.append(is_max_context)
                tokens.append(all_doc_tokens[split_token_idx])
                labels_ids.append(tags_to_ids[all_doc_labels[split_token_idx]])
                segment_ids.append(0)
      
                for feature, value in all_extra_features.items():
                    converted_label = map_features[feature][value[split_token_idx]]
                    extra_features_dic[feature] += [converted_label]
      
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            input_masks = [1] * len(input_ids)
      
            while len(input_ids) < doc_max_len:
                input_ids.append(0)
                input_masks.append(0)
                segment_ids.append(0)
                labels_ids.append(tags_to_ids['X'])
                tokens.append("X")
                token_is_max_context.append(False)
                for feature, value in all_extra_features.items():
                    extra_features_dic[feature] += [0]
      
      
            assert len(input_ids) == doc_max_len
            assert len(input_masks) == doc_max_len
            assert len(segment_ids) == doc_max_len
            assert len(labels_ids) == doc_max_len
            
            if extra_features_names != []:
                for doc_features in extra_features_dic.values():
                    assert len(doc_features) == doc_max_len
    
      
            span_features = dict()
            
            span_features['tokens'] = tokens
            span_features["input_ids"] = input_ids
            span_features["masks"] = input_masks
            span_features["segment_ids"] = segment_ids
            span_features["label_ids"] = labels_ids
            span_features['token_is_max_context'] = token_is_max_context
            span_features['doc_span_index'] = doc_span_index
            span_features['doc_index'] = doc_index
            span_features['prediction_masks'] = all_prediction_mask
            for feature, value in extra_features_dic.items():
              span_features[feature] = value
            
            all_docs_features.append(span_features)
          
    return all_docs_features
    