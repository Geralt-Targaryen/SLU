#coding=utf8
import os
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification


class SLUTagging_bert(nn.Module):

    def __init__(self, config, model):
        super(SLUTagging_bert, self).__init__()
        install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = config
        self.bert = AutoModelForTokenClassification.from_pretrained(model, cache_dir=os.path.join(install_path, 'cache'))
        self.output_layer = TaggingFNNDecoder(768, config.num_tags, config.tag_pad_idx)

        self.bert.classifier = nn.Identity()

    def forward(self, batch):
        tag_ids = batch.tag_ids                                 # (bs, len)
        tag_mask = batch.tag_mask                               # (bs, len)
        input_ids = batch.input_ids                             # (bs, len)
        attention_mask = batch.attention_mask                   # (bs, len)

        hiddens = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            ).logits                            # (bs, len, 768)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[1 : len(batch.utt[i])+1]            # remove the tag for [CLS]
            for idx, tid in enumerate(pred):                # for each token's pred in the sentence
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob,)
