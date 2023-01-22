import json
import os
from transformers import AutoTokenizer

from utils.vocab import Vocab
from utils.vocab_bert import LabelVocab
from utils.evaluator import Evaluator


class Example_bert_pretrain():

    @classmethod
    def configuration(cls, root, model, train_path=None):
        install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.evaluator = Evaluator()
        cls.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=os.path.join(install_path, 'cache'))
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.label_vocab = LabelVocab(root, cls.tokenizer)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:                        # one round in a multi-round dialogue
                if len(utt['asr_1best']) != len(utt['manual_transcript']): # ignore those data with uneven input output
                    continue
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example_bert_pretrain, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']                  # ASRed user utterance
        self.utt_denoised = ex['manual_transcript']
        self.input_idx = Example_bert_pretrain.tokenizer.convert_tokens_to_ids([c for c in self.utt])      # naive tokenization， 一个字一个字tokenize
        self.output_idx = [Example_bert_pretrain.word_vocab[c] for c in self.utt_denoised]
