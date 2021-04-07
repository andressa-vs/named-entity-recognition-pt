from lists import get_classifiers
import re


AGNOMES = ('jr','junior','primeiro','segundo','i','ii',
                        'filho','neto')

COMPANIES = ('ltda','sa','me','mei','epp','eireli')

MONTHS = ('janeiro', 'fevereiro','março','abril','maio','junho','julho',
         'agosto','setembro','outubro','novembro','dezembro')


def ort_lex_features(ne):
    last_en_token = ne.split()[-1]
    map_months = lambda x: x in MONTHS
    is_date = list(map(map_months, ne.split()))
    
    if "&" in ne or last_en_token in COMPANIES:
        return 'ORGANIZAÇÃO'
    if last_en_token in AGNOMES:
        return 'PESSOA'
    if re.search('\$|€|¥|£|₤|%', ne):
        return 'VALOR'
    if True in is_date:
        return "TEMPO"
    return 'O'

def match_contexts(first_index, last_index, sentence, scenario):
        
    first_en_token = sentence[first_index].lower() 
    prevWord = sentence[first_index-1].lower() if first_index-1 > 0 else 'NONE'
    ne = ' '.join(token.lower() for token in sentence[first_index:last_index])
        
    if scenario == 'total':
        classifiers = get_classifiers()
    elif scenario == 'selective':
        classifiers = get_classifiers(['PESSOA','LOCAL','ORGANIZAÇÃO'])
    else:
        raise ValueError('Scenario must be "total" or "selective".')
        
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
            if document[prev_index] in ("e", ","):
                sequence += 1
                if document[prev_index] == "e":
                    e_bool = True
                    break 
        else:
            break
        prev_index = last_index
    
    if e_bool:
        return sequence
    
    return 0
        
    
def classify_contexts(document, doc_entities, scenario):
    
    context_features = ['O'] * len(document)
       
    sequence = 0
    
    for i, (first_index, last_index) in enumerate(doc_entities):
        if sequence == 0:
            clas, ft = match_contexts(first_index,last_index,document,scenario)
        
        if clas != 'O':
            ne_len = last_index - first_index
            context_features[first_index:last_index] = [clas] * ne_len   
            
        if ft == 'pl':
            sequence = is_sequence(document,last_index,doc_entities[i+1:])
            
    return context_features
