#-*- coding:utf-8 -*-
import torch


def from_example_list_pretrain(args, ex_list, device='cpu', train=True):
    # ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    output_pad_idx = 0 # pad as a tag for paddings

    batch.utt = [ex.utt for ex in ex_list] # list of str
    input_lens = [len(ex.input_idx) for ex in ex_list] # list of length for each input
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens

    assert train == True

    output_lens = [len(ex.output_idx) for ex in ex_list]
    max_output_lens = max(output_lens)
    output_ids = [ex.output_idx + [output_pad_idx] * (max_output_lens - len(ex.output_idx)) for ex in ex_list]
    output_mask = [[1] * len(ex.output_idx) + [0] * (max_output_lens - len(ex.output_idx)) for ex in ex_list]
    batch.output_ids = torch.tensor(output_ids, dtype=torch.long, device=device)
    batch.output_mask = torch.tensor(output_mask, dtype=torch.float, device=device)

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]