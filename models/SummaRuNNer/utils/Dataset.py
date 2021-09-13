import csv
import torch
import torch.utils.data as data
from torch.autograd import Variable
from .Vocab import Vocab
from pathlib import Path
from utils.tokenizer import Tokenizer
import numpy as np
import re

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', '！', '。', '？',
              dm_single_close_quote, dm_double_close_quote, ")"]
PAD_TOKEN = '<PAD>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '<UNK>' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<START>' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<END>' # This has a vocab id, which is used at the end of untruncated target sequences

class Dataset(data.Dataset):
    def __init__(self, features):
        self.data = features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " 。"

def punct_tras(line):
    line = re.sub(',', '，', line)
    line = re.sub('！', '!', line)
    line = re.sub('？', '?', line)
    return line

def binary_label(labels, length):
    b_labels = []
    for i in range(length):
        if i in labels:
            b_labels.append(1)
        else:
            b_labels.append(0)
    return b_labels

def binary_label_single(labels, context, role):
    b_labels = []
    for i, turn in enumerate(context):
        if turn['speaker'] == role:
            if i in labels:
                b_labels.append(1)
            else:
                b_labels.append(0)
    return b_labels

def get_labels(sample, context_mode):
    user_labels, agent_labels = [], []
    for qa in sample['QA']:
        user_labels.extend(qa['QueSummUttIDs'])
        agent_labels.extend(qa['AnsSummLongUttIDs'])
    if context_mode == 'both':
        b_user_labels = binary_label(list(set(user_labels)), len(sample['Dialogue']))
        b_agent_labels = binary_label(list(set(agent_labels)), len(sample['Dialogue']))
    else:
        b_user_labels = binary_label_single(list(set(user_labels)), sample['Dialogue'], 'Q')
        b_agent_labels = binary_label_single(list(set(agent_labels)), sample['Dialogue'], 'A')
    b_final_labels = binary_label(list(set(user_labels + agent_labels)), len(sample['Dialogue']))
    return b_user_labels, b_agent_labels, b_final_labels

def convert_example_to_feature(args, data, org_vocab, get_vocab):
    features = []
    num_len_below, num_len_over = 0, 0
    sums, labels, contexts, doc_lens = [], [], [], []
    for sample in data:
        user_labels, agent_labels, final_labels = get_labels(sample, args.context_mode)
        if args.sum_mode == 'final':
            label = final_labels
            sum = sample['FinalSumm']
        elif args.sum_mode == 'user':
            label = user_labels
            sum = sample['UserSumm']
        elif args.sum_mode == 'agent':
            label = agent_labels
            sum = sample['AgentSumm']
        labels.append(label[:args.doc_trunc])
        sums.append(sum)

        context = []
        for turn in sample['Dialogue']:
            tmp_utt = []
            if turn['speaker'] == 'Q':
                tmp_utt += [sample['QRole'], ':']
            else:
                tmp_utt += ['客服', ':']
            for word in turn['utterance'].split():
                if len(word) > 2 and word[0] == '[' and word[-1] == ']':
                    tmp_utt += ['[', word[1:-1], ']']
                else:
                    tmp_utt.append(word)
            if len(tmp_utt) > args.sent_trunc:
                num_len_over += 1
            else:
                num_len_below += 1
            tmp_utt = ' '.join(tmp_utt)
            tmp_utt = fix_missing_period(tmp_utt)
            tmp_utt = punct_tras(tmp_utt)
            if args.context_mode == 'both':
                context.append(tmp_utt)
            elif args.context_mode == 'user' and turn['speaker'] == 'Q':
                context.append(tmp_utt)
            elif args.context_mode == 'agent' and turn['speaker'] == 'A':
                context.append(tmp_utt)
        context = context[:args.doc_trunc]
        doc_lens.append(len(context))
        contexts.append(context)

    # get vocab
    if get_vocab:
        if args.new_vocab:
            get_dialogue_vocab([], contexts, Path('data/embeddings/'), 'word')
            dialogue_vocab = load_vocab(Path('data/embeddings/dialogue_vocab_word'))
            get_tencent_embedding(dialogue_vocab, Path('/home/sdb/htlin/Resource/Tencent_AILab_ChineseEmbedding.txt'),
                                  Path(args.vocab_path))
        vocab = Tokenizer(args.vocab_path, args.embed_dim, args.vocab_size)
    else:
        vocab = org_vocab

    for i, context in enumerate(contexts):
        context_ids = [vocab.tokenize(s, True)[:args.sent_trunc] for s in context]
        context = [re.sub(' ', '', s) for s in context]

        features.append(
            {'features': context_ids,
             'targets': labels[i],
             'summaries': sums[i],
             'contexts': context,
             'doc_lens': doc_lens[i],
             'pad': vocab.token2idx(PAD_TOKEN)}
            )
    print(num_len_below, num_len_over)

    return features, vocab

def padded_sequence(seqs, pad):
    max_len = max([len(seq) for seq in seqs])
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    length = [len(seq) for seq in seqs]
    return padded_seqs, length

def batchify_data(batch):
    features = [f['features'] for f in batch]
    features = [s for sample in features for s in sample]
    features, _ = padded_sequence(features, batch[0]['pad'])
    features = torch.tensor(features, dtype=torch.long)

    targets = [f['targets'] for f in batch]
    targets = [s for sample in targets for s in sample]
    targets = torch.tensor(targets, dtype=torch.long)

    doc_lens = [f['doc_lens'] for f in batch]
    doc_lens = torch.tensor(doc_lens, dtype=torch.long)
    summaries = [f['summaries'] for f in batch]
    docs = [f['contexts'] for f in batch]

    return features, targets, summaries, doc_lens, docs

def get_train_dataloader(args, train_data):
    print('processing training data-------------')
    features, vocab = convert_example_to_feature(args, train_data, None, get_vocab=True)
    train_data = Dataset(features)
    sampler = data.RandomSampler(train_data)
    train_dataloader = data.DataLoader(train_data, sampler=sampler, batch_size=args.batch_size, collate_fn=batchify_data)
    return train_dataloader, features, vocab

def get_val_dataloader(args, val_data, vocab):
    print('processing val data-------------')
    features, vocab = convert_example_to_feature(args, val_data, vocab, get_vocab=False)
    # if mode == 'eval':
    val_data = Dataset(features)
    sampler = data.SequentialSampler(val_data)
    val_dataloader = data.DataLoader(val_data, sampler=sampler, batch_size=args.batch_size, collate_fn=batchify_data)
    # else:
    #     decode_features = []
    #     for f in features:
    #         decode_features.extend([f] * beam_size)
    #     val_data = PGNDataset(decode_features)
    #     sampler = SequentialSampler(val_data)
    #     val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=beam_size, collate_fn=batchify_data)
    return val_dataloader, features

def get_test_dataloader(args, test_data):
    print('processing test data-------------')
    features, vocab = convert_example_to_feature(args, test_data, None, get_vocab=True)
    # if mode == 'eval':
    test_data = Dataset(features)
    sampler = data.SequentialSampler(test_data)
    test_dataloader = data.DataLoader(test_data, sampler=sampler, batch_size=args.batch_size, collate_fn=batchify_data)
    # else:
    #     decode_features = []
    #     for f in features:
    #         decode_features.extend([f] * beam_size)
    #     val_data = PGNDataset(decode_features)
    #     sampler = SequentialSampler(val_data)
    #     val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=beam_size, collate_fn=batchify_data)
    return test_dataloader, features, vocab
