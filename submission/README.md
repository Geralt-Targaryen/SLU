## Code Structure

+ Common utilities: `utils/initialization.py`, `utils/args.py`, `utils/evaluator.py`
+ LSTM baseline
    + main: `scripts/slu_baseline.py`
    + model: `model/slu_baseline_tagging.py`
    + utililities: `utils/example.py`, `utils/batch.py`
    + vocabulary: `utils/vocab.py`, `utils/word2vec.py`, embeddings loaded from `word2vec-768.txt`
+ Denoising LSTM
  + main: `scripts/slu_baseline_pretrain.py`
  + model: `model/slu_baseline_tagging_pretrain.py`
  + utililities: `utils/example_pretrain.py`, `utils/batch_pretrain.py`, `utils/example.py`, `utils/batch.py`
  + vocabulary: `utils/vocab.py`, `utils/word2vec.py`, embeddings loaded from `word2vec-768.txt`
+ Vanilla RoBERTa & XLM-R
  + main: `scripts/roberta.py`
  + model: `model/roberta_tagging.py`
  + utilities: `utils/example_bert.py`, `utils/batch_bert.py`, `utils/vocab_bert.py`
+ Denoising RoBERTa & XLM-R
  + main: `scripts/roberta_pretrain.py`
  + model: `model/roberta_tagging_pretrain.py`
  + utilities: `utils/example_bert_pretrain.py`, `utils/batch_bert_pretrain.py`, `utils/example_bert.py`, `utils/batch_bert.py`, `utils/vocab_bert.py`
+ mT5
  + main: `scripts/mt5.py`
  + model: `model/mt5_seq2seq.py`
  + utilities: `utils/example_t5.py`, `utils/batch_t5.py`, `utils/vocab_t5.py`
+ misc
  + `logs` directory: all our training logs
  + `data` directory: input data and output predictions
  + `README.md`: this file
  + `requirements.txt`: python environments

## Setup
Python version: 3.7

    pip install -r requirements.txt

## Run
Note: all our training is conducted on an RTX 3090 with 24GB memory. If you encounter `cuda out of memory`, especially when training mT5, try a smaller batch size.

### LSTM
#### baseline
train：

    python scripts/slu_baseline.py --device 0 --max_epoch 20
evaluate (on the development set) and predict (on the test set)：

    python scripts/slu_baseline.py --device 0 --testing --checkpoint model.bin

#### dual-branch denoising
    python scripts/slu_baseline_pretrain.py --device 0 --max_epoch 20 --checkpoint model_pretrained.pth
    python scripts/slu_baseline_pretrain.py --device 0 --testing --checkpoint model_pretrained.pth

### Chinese RoBERTa
#### vanilla
    python scripts/roberta.py --device 0 --max_epoch 20 --lr 1e-5 --checkpoint model_roberta.pth
    python scripts/roberta.py --device 0 --testing --checkpoint model_roberta.pth

#### dual-branch denoising
    python scripts/roberta_pretrain.py --device 0 --max_epoch 20 --lr 1e-5 --checkpoint model_roberta_pretrained.pth
    python scripts/roberta_pretrain.py --device 0 --testing --checkpoint model_roberta_pretrained.pth

### XLM-R
#### vanilla
    python scripts/roberta.py --device 0 --max_epoch 20 --model_name_or_path xlm-roberta-base --lr 1e-5 --checkpoint model_xlmr.pth
    python scripts/roberta.py --device 0 --testing --model_name_or_path xlm-roberta-base --checkpoint model_xlmr.pth

#### dual-branch denoising
    python scripts/roberta_pretrain.py --device 0 --max_epoch 20 --model_name_or_path xlm-roberta-base --lr 1e-5 --checkpoint model_xlmr_pretrained.pth
    python scripts/roberta_pretrain.py --device 0 --testing --model_name_or_path xlm-roberta-base --checkpoint model_xlmr_pretrained.pth

### mT5
    python scripts/mt5.py --device 0 --max_epoch 125 --lr 1e-4 --checkpoint model_mt5.pth
    python scripts/mt5.py --device 0 --testing --checkpoint model_mt5.pth

## Inference
To run inference on other test data, replace `data/test_unlabelled.json`.



