"""produce the dataset with (psudo) extraction label"""
import os
import jieba
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose

from utils import count_data
from metric import compute_rouge_l
import argparse



try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    DATA_DIR = 'dataset'


def _split_words(texts):
    return map(lambda t: t.split(), texts)

def add_period(sent):
    sent = sent.strip()
    if sent[-1] != '。' and sent[-1] != '？' and sent[-1] != '?':
        return sent + ' 。'
    else:
        return sent


def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(compute_rouge_l(reference=abst, mode='r'),
                          art_sents))
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores

def convert_example_to_feature(data, args):
    sums, contexts, turns = [], [], []
    for sample in data:
        if args.sum_mode == 'final':
            sum = sample['FinalSumm'] # list
        elif args.sum_mode == 'user':
            sum = sample['UserSumm']
        elif args.sum_mode == 'agent':
            sum = sample['AgentSumm']
        # consider , as a sentence
        if args.split_mode == 'comma':
            split_sum = []
            for s in sum:
                last_index = 0
                for i in range(len(s)):
                    if s[i] == '，' or s[i] == '。' or s[i] == ',':
                        split_sum.append(s[last_index:i+1])
                        last_index = i+1
            split_sum = [' '.join(jieba.lcut(s)) for s in split_sum]
            sums.append(split_sum)
        elif args.split_mode == 'period':
            # split_sum = []
            # for s in sum:
            #     last_index = 0
            #     for i in range(len(s)):
            #         if s[i] in ['。', '.', '!', '！', '?', '？']:
            #             split_sum.append(s[last_index:i + 1])
            #             last_index = i + 1
            #     if last_index != len(s):
            #         split_sum.append(s[last_index:])
            split_sum = [' '.join(jieba.lcut(s)) for s in sum]
            tmp_sums = []
            for sum in split_sum:
                if sum.strip() != '':
                    tmp_sums.append(sum)
            sums.append(tmp_sums)

        context = []
        if args.turn_mode == 'single':
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
                tmp_utt = ' '.join(tmp_utt)
                if args.context_mode == 'both':
                    context.append(tmp_utt)
                elif args.context_mode == 'user' and turn['speaker'] == 'Q':
                    context.append(tmp_utt)
                elif args.context_mode == 'agent' and turn['speaker'] == 'A':
                    context.append(tmp_utt)
        elif args.turn_mode == 'multi':
            last_speaker, tmp_utt = '', []
            for turn in sample['Dialogue']:
                turn['utterance'] = add_period(turn['utterance'])
                if last_speaker != turn['speaker']:
                    if tmp_utt != []:
                        if args.context_mode == 'both':
                            context.append(' '.join(tmp_utt))
                        elif args.context_mode == 'user' and last_speaker == 'Q':
                            context.append(' '.join(tmp_utt))
                        elif args.context_mode == 'agent' and last_speaker == 'A':
                            context.append(' '.join(tmp_utt))
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
                    last_speaker = turn['speaker']
                else:
                    for word in turn['utterance'].split():
                        if len(word) > 2 and word[0] == '[' and word[-1] == ']':
                            tmp_utt += ['[', word[1:-1], ']']
                        else:
                            tmp_utt.append(word)
            if args.context_mode == 'both':
                context.append(' '.join(tmp_utt))
            elif args.context_mode == 'user' and last_speaker == 'Q':
                context.append(' '.join(tmp_utt))
            elif args.context_mode == 'agent' and last_speaker == 'A':
                context.append(' '.join(tmp_utt))
        contexts.append(context)
    return sums, contexts

def label(split, args):
    start = time()
    print('start processing {} split...'.format(split))
    data_path = join(DATA_DIR, split)
    data_dir = data_path[:-5] + '/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(data_path, 'r') as f:
        data = json.load(f)
    sums, contexts = convert_example_to_feature(data, args)
    for i, (sum, context) in enumerate(zip(sums, contexts)):
        extracted, scores = get_extract_label(context, sum)
        sample = {'article': context,
                  'abstract': sum}
        sample['extracted'] = extracted
        sample['score'] = scores
        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(sample, f, indent=4, ensure_ascii=False)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(args):
    train_file = 'train_augmented.json' if args.augment else 'train.json'
    for split in ['val.json', train_file]:  # no need of extraction label when testing
        label(split, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_mode', required=True, type=str)
    parser.add_argument('--turn_mode', required=True, type=str)
    parser.add_argument('--sum_mode', required=True, type=str)
    parser.add_argument('--context_mode', required=True, type=str)
    parser.add_argument("--augment", action='store_true')
    args = parser.parse_args()
    main(args)
