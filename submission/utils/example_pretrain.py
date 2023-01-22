import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

# from utils.vocab_pretrain import Vocab_pretrain, LabelVocab
class Example_pretrain():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        # train_path: train.json 's path
        cls.evaluator = Evaluator() 
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:                        # one round in a multi-round dialogue
                if len(utt['asr_1best']) != len(utt['manual_transcript']): # ignore those data with uneven input output
                    continue
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example_pretrain, self).__init__()

        self.ex = ex

        self.utt = ex['asr_1best']                  # ASRed user utterance
        self.utt_denoised = ex['manual_transcript']
        self.input_idx = [Example_pretrain.word_vocab[c] for c in self.utt]      # naive tokenization， 一个字一个字tokenize
        self.output_idx = [Example_pretrain.word_vocab[c] for c in self.utt_denoised]
