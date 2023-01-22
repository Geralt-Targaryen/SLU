import json
import os
from transformers import AutoTokenizer

from utils.vocab_bert import LabelVocab
from utils.evaluator import Evaluator


class Example_bert():

    @classmethod
    def configuration(cls, root, model):
        install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.evaluator = Evaluator()
        cls.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=os.path.join(install_path, 'cache'))
        cls.label_vocab = LabelVocab(root, cls.tokenizer)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:                        # one round in a multi-round dialogue
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example_bert, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']                  # ASRed user utterance
        self.slot = {}
        if ex.get('semantic', None) is not None:
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)           # tags: list[str], same length as raw sentence
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:                          # if the labeled value is in the sentence
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = Example_bert.tokenizer.convert_tokens_to_ids([c for c in self.utt])     # naive tokenization
        l = Example_bert.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]