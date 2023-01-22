#-*- coding:utf-8 -*-
import torch


def from_example_list(args, ex_list, device='cpu', train=True):
    EOS_IDX = args.tokenizer.eos_token_id
    PAD_IDX = args.tokenizer.pad_token_id

    batch = Batch_t5(ex_list, device)

    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) + 1 for ex in ex_list]          # + 1 for </s>
    max_len = max(input_lens)
    batch.max_len = max_len
    input_ids = [ex.input_idx + [EOS_IDX] + [PAD_IDX] * (max_len - len(ex.input_idx) - 1) for ex in ex_list]
    attention_mask = [[1] * (len(ex.input_idx) + 1) + [0] * (max_len - len(ex.input_idx) - 1) for ex in ex_list]
    # print(batch.utt)
    # print(input_ids)
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.output_idx) + 2 for ex in ex_list]       # + 1 for <pad> and +1 for </s>
        max_tag_lens = max(tag_lens)
        output_ids = [[PAD_IDX] + ex.output_idx + [EOS_IDX] + [PAD_IDX] * (max_tag_lens - len(ex.output_idx) - 2) for ex in ex_list]
        output_mask = [[1] * (len(ex.output_idx) + 2) + [0] * (max_tag_lens - len(ex.output_idx) - 2) for ex in ex_list]
        batch.output_ids = torch.tensor(output_ids, dtype=torch.long, device=device)
        batch.output_mask = torch.tensor(output_mask, dtype=torch.float, device=device)
        # print(batch.labels)
        # print(batch.output_ids)
        # print(batch.output_mask)
    else:
        batch.labels = None
        batch.output_ids = None
        batch.output_mask = None

    return batch


class Batch_t5():

    def __init__(self, examples, device):
        super(Batch_t5, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]