#coding=utf8
import os
import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration


class SLUTagging_t5(nn.Module):

    def __init__(self, config, model):
        super(SLUTagging_t5, self).__init__()
        install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = config
        self.tokenizer = config.tokenizer
        self.t5: MT5ForConditionalGeneration = MT5ForConditionalGeneration.from_pretrained(model, cache_dir=os.path.join(install_path, 'cache'))
        # adjust the vocabulary
        self.t5.resize_token_embeddings(len(self.tokenizer))
        #self.tokenizer.all_special_ids) and i != self.tokenizer.convert_tokens_to_ids('O'):
        bad_words_ids = list(range(len(self.tokenizer)))
        for i in self.tokenizer.all_special_ids + [self.tokenizer.convert_tokens_to_ids('O')]:
            bad_words_ids.remove(i)
        self.bad_words_ids = [[id] for id in bad_words_ids]

    def forward(self, batch):
        input_ids = batch.input_ids                             # (bs, len)
        attention_mask = batch.attention_mask                   # (bs, len)

        if batch.output_ids is not None:
            output_ids = batch.output_ids[:, :-1].contiguous()
            labels = batch.output_ids[:, 1:].clone().detach()
            labels[batch.output_ids[:, 1:] == self.tokenizer.pad_token_id] = -100
        else:
            output_ids, labels = None, None

        loss = self.t5(input_ids=input_ids, 
                       attention_mask=attention_mask,
                       decoder_input_ids=output_ids,
                       labels=labels)[0]
        # print(loss)
        return loss

    def decode(self, batch):
        batch_size = len(batch)
        labels = batch.labels
        loss = None
        with torch.no_grad():
            if labels is not None:
                loss = self.forward(batch)
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            pred_ids = self.t5.generate(inputs=input_ids,
                                        attention_mask=attention_mask,
                                        max_length=batch.max_len,
                                        bad_words_ids=self.bad_words_ids)
        
        predictions = []
        for i in range(batch_size):
            pred_id = pred_ids[i]
            pred = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in pred_id]
            # tag_id = batch.output_ids[i]
            # target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in tag_id]
            # print(batch.utt[i])
            # print(input_ids[i])
            # print('target')
            # print(target)
            # print('pred')
            # print(pred)

            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[1 : len(batch.utt[i])+1]            # remove the tag for <pad>
            # print(pred, end='\n\n')
            for idx, tag in enumerate(pred):                # for each token's pred in the sentence
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
            # print(pred_tuple)
            if any([l.split('-')[-1] not in batch.utt[i] for l in pred_tuple]):
                print(batch.utt[i])
                print(input_ids[i])
                print(pred_id)
                print(pred)
                print(pred_tuple)
            predictions.append(pred_tuple)

        if labels is None:
            return predictions
        else:
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
