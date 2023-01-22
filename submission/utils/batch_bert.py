#-*- coding:utf-8 -*-
import torch


def from_example_list(args, ex_list, device='cpu', train=True):
    CLS_IDX = args.tokenizer.cls_token_id
    SEP_IDX = args.tokenizer.sep_token_id
    PAD_IDX = args.tokenizer.pad_token_id

    batch = Batch_bert(ex_list, device)
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) + 2 for ex in ex_list]          # 1 for cls and 1 for sep
    max_len = max(input_lens)
    input_ids = [[CLS_IDX] +  ex.input_idx + [SEP_IDX] + [PAD_IDX] * (max_len - len(ex.input_idx) - 2) for ex in ex_list]
    attention_mask = [[1] * (len(ex.input_idx) + 2) + [0] * (max_len - len(ex.input_idx) - 2) for ex in ex_list]
    # print(batch.utt)
    # print(input_ids)
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) + 2 for ex in ex_list]           # 1 for cls and 1 for sep
        max_tag_lens = max(tag_lens)
        tag_ids = [[tag_pad_idx] + ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id) - 1) for ex in ex_list]
        tag_mask = [[0] + [1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id) - 1) for ex in ex_list]
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
        # print(batch.labels)
        # print(batch.tag_ids)
        # print(batch.tag_mask)
    else:
        batch.labels = None
        batch.tag_ids = None
        tag_mask = [[0] + [1] * len(ex.input_idx) + [0] * (max_len - len(ex.input_idx) - 1) for  ex in ex_list]
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)

    return batch


class Batch_bert():

    def __init__(self, examples, device):
        super(Batch_bert, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]