#coding=utf8
import sys, os, time, gc, json
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD


from model.slu_baseline_tagging_pretrain import SLUTagging
from utils.example_pretrain import Example_pretrain
from utils.batch_pretrain import from_example_list_pretrain

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)

Example_pretrain.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)


train_dataset = Example.load_dataset(train_path)                                                # list[Example]
dev_dataset = Example.load_dataset(dev_path)

pretrain_dataset = Example_pretrain.load_dataset(train_path)

print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size                     # baseline: 1741
args.pad_idx = Example.word_vocab[PAD]                              # baseline:  0
args.num_tags = Example.label_vocab.num_tags                        # baseline: 74
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)      # baseline: 0

model: SLUTagging = SLUTagging(args).to(device)
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)



def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch) # pred is in the form of act-slot-value
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)

    return metrics, total_loss / count


def predict():
    model.eval()
    test_path = os.path.join(args.dataroot,  'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            predictions += pred
    test_json = json.load(open(test_path, 'r'))
    ptr = 0
    for example in test_json:
        for utt in example:
            utt['pred'] = [pred.split('-') for pred in predictions[ptr]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w'), indent=4, ensure_ascii=False)





if not args.testing:
    optimizer = set_optimizer(model, args)
    """ 
    pretraining
    """
    num_pretrain_epochs = 10
    nsamples_pretrain = len(pretrain_dataset)
    pretrain_index, step_size = np.arange(nsamples_pretrain), args.batch_size
    print('Start pre-training ......')
    tic = time.time()
    for i in range(num_pretrain_epochs):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(pretrain_index)
        model.train()
        count = 0
        for j in range(0, nsamples_pretrain, step_size):
            cur_dataset = [pretrain_dataset[k] for k in pretrain_index[j: j + step_size]]
            current_batch = from_example_list_pretrain(args, cur_dataset, device, train=True)
            output, loss = model.pretrain(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('Pre-training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))

    """ 
    training
    """
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0., 'dev_loss': 1000}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open(args.checkpoint, 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    print('TOTAL TRAINING TIME: %.2f' % (time.time() - tic))

else:
    assert os.path.exists(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
