#coding=utf8
import os, json


def LabelVocab_t5(root, tokenizer):
    ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r'))
    acts = ontology['acts']
    slots = ontology['slots']

    additional_special_tokens = []

    for act in acts:
        for slot in slots:
            for bi in ['B', 'I']:
                tag = f'{bi}-{act}-{slot}'
                additional_special_tokens.append(tag)
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    print(f'added {num_added} special tokens to {type(tokenizer)}')
