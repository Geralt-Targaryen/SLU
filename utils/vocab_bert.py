#coding=utf8
import os, json


class LabelVocab():

    def __init__(self, root, tokenizer):
        self.tag2idx, self.idx2tag = {}, {}

        self.tag2idx[tokenizer.pad_token] = 0
        self.idx2tag[0] = tokenizer.pad_token
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r'))
        acts = ontology['acts']
        slots = ontology['slots']

        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)
