#!/usr/bin/env python3

import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm

# added
from utils.Dataset import get_train_dataloader, get_val_dataloader, get_test_dataloader
import re
import files2rouge

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_path',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=200)
parser.add_argument('-embed_num',type=int,default=500)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=50)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=10)
parser.add_argument('-seed',type=int,default=2020)
parser.add_argument('-train_dir',type=str,default='data/train.json')
parser.add_argument('-val_dir',type=str,default='data/val.json')
parser.add_argument('-vocab_path',type=str,default='data/embeddings/dialogue_embed_word')
parser.add_argument('-vocab_size',type=int,default=10000)
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-log_freq',type=int,default=100)
parser.add_argument('-report_every',type=int,default=2000)
parser.add_argument('-sent_trunc',type=int,default=100)
parser.add_argument('-doc_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_2020.pt')
parser.add_argument('-test_dir',type=str,default='data/test.json')
parser.add_argument('-output_dir',type=str,default='outputs/')
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=15)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
# add
parser.add_argument('-sum_mode',type=str,default='final')
parser.add_argument('-context_mode',type=str,default='both')
parser.add_argument('-new_vocab',type=bool,default=False)
parser.add_argument('-max_sum',type=int,default=100)
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 
    
def eval(net,vocab,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        features,targets,_,doc_lens, _ = batch
        features,targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features,doc_lens)
        loss = criterion(probs,targets)
        total_loss += loss.item()
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss

def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')
    
    # embed = torch.Tensor(np.load(args.embedding)['embedding'])
    # with open(args.word2id) as f:
    #     word2id = json.load(f)
    # vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir, 'r') as f:
        train_data = json.load(f)
    train_iter, _, vocab = get_train_dataloader(args, train_data)

    with open(args.val_dir, 'r') as f:
        val_data = json.load(f)
    val_iter, _ = get_val_dataloader(args, val_data, vocab)

    # update args
    embed = torch.tensor(vocab.embedding, dtype=torch.float)
    args.embed_num = embed.shape[0]
    args.embed_dim = embed.shape[1]
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    net = getattr(models,args.model)(args, embed)
    if use_gpu:
        net.cuda()
    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    
    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    net.train()
    
    t1 = time()
    step = 0
    for epoch in range(1,args.epochs+1):
        for i,batch in enumerate(train_iter):
            features,targets,_,doc_lens, _ = batch
            features,targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features,doc_lens)
            loss = criterion(probs,targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()
            step += 1
            if step % args.log_freq == 0:
                logging.info('Batch ID:%d Loss:%f' % (i,loss.item()))
        cur_loss = eval(net,vocab,val_iter,criterion)
        if cur_loss < min_loss:
            min_loss = cur_loss
            best_path = net.save()
        logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                % (epoch,min_loss,cur_loss))
    t2 = time()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))

def get_label_index(label):
    index = []
    label = label.cpu().numpy().tolist()
    for i in range(len(label)):
        if label[i] == 1:
            index.append(i)
    return index

def cal_rouge(refs, preds, ref_name, pred_name):
    ref_ids, pred_ids = [], []
    for ref, pred in zip(refs, preds):
        ref = re.sub('\n', '', ref)
        pred = re.sub('\n', '', pred)
        ref_id, pred_id = [], []
        tmp_dict = {}
        new_index = 0
        words = list(ref)
        for w in words:
            if w not in tmp_dict.keys():
                tmp_dict[w] = new_index
                ref_id.append(str(new_index))
                new_index += 1
            else:
                ref_id.append(str(tmp_dict[w]))
        words = list(pred)
        for w in words:
            if w not in tmp_dict.keys():
                tmp_dict[w] = new_index
                pred_id.append(str(new_index))
                new_index += 1
            else:
                pred_id.append(str(tmp_dict[w]))
        ref_ids.append(' '.join(ref_id))
        pred_ids.append(' '.join(pred_id))
    with open(ref_name, 'w') as f:
        for ref in ref_ids:
            f.write(ref + '\n')
    with open(pred_name, 'w') as f:
        for pred in pred_ids:
            f.write(pred + '\n')
    files2rouge.run(pred_name, ref_name)

def test():
     
    # embed = torch.Tensor(np.load(args.embedding)['embedding'])
    # with open(args.word2id) as f:
    #     word2id = json.load(f)
    # vocab = utils.Vocab(embed, word2id)

    with open(args.test_dir, 'r') as f:
        test_data = json.load(f)
    test_iter, _, vocab = get_test_dataloader(args, test_data)

    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(test_data)
    time_cost = 0
    refs, hyps, ext_golds = [], [], []
    for batch in tqdm(test_iter):
        features, labels, summaries, doc_lens, docs = batch
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            word_count = 0
            topk_chosen = []
            for ind in topk_indices:
                word_count += len(docs[doc_id][ind])
                if word_count > args.max_sum and topk_chosen:
                    break
                else:
                    topk_chosen.append(ind)
            topk_chosen = np.array(topk_chosen)
            topk_chosen.sort()
            doc = docs[doc_id][:doc_len]
            hyp = [doc[index] for index in topk_chosen]
            oracle = [doc[index] for index in get_label_index(labels)]
            ref = summaries[doc_id]
            refs.append(re.sub(' ', '', '\n'.join(ref)))
            hyps.append(re.sub(' ', '', '\n'.join(hyp)))
            ext_golds.append(re.sub(' ', '', '\n'.join(oracle)))
            start = stop
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.ref = args.output_dir + 'ref.txt'
    args.hyp = args.output_dir + 'hyp.txt'
    args.oracle = args.output_dir + 'oracle.txt'
    with open(os.path.join(args.ref), 'w') as f:
        for ref in refs:
            f.write(ref + '\n\n')
    with open(os.path.join(args.hyp), 'w') as f:
        for hyp in hyps:
            f.write(hyp + '\n\n')
    with open(os.path.join(args.oracle), 'w') as f:
        for s in ext_golds:
            f.write(s + '\n\n')
    with open(os.path.join(args.ref[:-4] + '_cal.txt'), 'w') as f:
        for ref in refs:
            ref = re.sub('\n', '', ref)
            f.write(ref + '\n')
    with open(os.path.join(args.hyp[:-4] + '_cal.txt'), 'w') as f:
        for hyp in hyps:
            hyp = re.sub('\n', '', hyp)
            f.write(hyp + '\n')
    with open(os.path.join(args.oracle[:-4] + '_cal.txt'), 'w') as f:
        for s in ext_golds:
            s = re.sub('\n', '', s)
            f.write(s + '\n')
    # run files2rouge
    print('reference and preds--------------------------')
    cal_rouge(refs, hyps, args.ref[:-4] + '_id.txt', args.hyp[:-4] + '_id.txt')
    print('preds and oracle--------------------------')
    cal_rouge(ext_golds, hyps, args.oracle[:-4] + '_id.txt', args.hyp[:-4] + '_id.txt')
    print('oracle and reference--------------------------')
    cal_rouge(refs, ext_golds, args.ref[:-4] + '_id.txt', args.oracle[:-4] + '_id.txt')
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


def predict(examples):
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)
    pred_dataset = utils.Dataset(examples)

    pred_iter = DataLoader(dataset=pred_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(pred_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(pred_iter):
        features, doc_lens = vocab.make_predict_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch[doc_id].split('. ')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('. '.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))

if __name__=='__main__':
    if args.test:
        test()
    elif args.predict:
        with open(args.filename) as file:
            bod = [file.read()]
        predict(bod)
    else:
        train()
