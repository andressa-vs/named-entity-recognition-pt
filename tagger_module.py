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
           'COISA','ABSTRAÇÃO', 'ACONTECIMENTO', 'OBRA']

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
        inside_class = 'I-{}'.format(entity_class[:3])
        entities_tags.extend((begin_class, inside_class))
        
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

def features(features_names, scenario):
        
    def feature_to_index(features):
        return {label:index for index, label in enumerate(features)}
    
    def one_hot_index(features):
        dic = {}
        for index, feature in enumerate(features):
            list_len = [0] * len(features)
            list_len[index] = 1
            dic[feature] = list_len
            
        return dic
    
    dic_map_features = {}
    x = ['X']
    o = ['O']

    if "ortographic_features" in features_names:
        type_cats = x + WORD_TYPE_TAGS
        case_cats = x + WORD_CASE_TAGS
        
        dic_map_features['wordTypeFeature'] = feature_to_index(type_cats)
        dic_map_features['wordCaseFeature'] = feature_to_index(case_cats)
    
    if "pos_tag_feature" in features_names:
        pos_cats = x + POS_TAGS
        dic_map_features['posTagFeature'] = feature_to_index(pos_cats) 
    
    if "word_context_feature" in features_names:
        out_feats = x + o
        if scenario == 'total':
            classes = out_feats + TOTAL_CLASSES
        elif scenario == 'selective':
            classes = out_feats + SELECTIVE_CLASSES 
      
        dic_map_features["wordContextFeature"] = one_hot_index(classes)
        
    return dic_map_features
            
            
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
        
