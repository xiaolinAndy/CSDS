""" make reference text files needed for ROUGE evaluation """
import json
import os
import jieba
from os.path import join, exists
from time import time
from datetime import timedelta

from utils import count_data
from decoding import make_html_safe
import argparse

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    DATA_DIR = 'dataset'

def add_period(sent):
    sent = sent.strip()
    if sent[-1] != '。' and sent[-1] != '？' and sent[-1] != '?':
        return sent + ' 。'
    else:
        return sent

def convert_example_to_feature(data, args):
    sums, contexts = [], []
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

def dump(split, args):
    start = time()
    print('start processing {} split...'.format(split))
    data_path = join(DATA_DIR, split)
    data_dir = join(DATA_DIR, 'test/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dump_dir = join(DATA_DIR, 'refs.txt')
    with open(data_path, 'r') as f:
        data = json.load(f)
    sums, contexts = convert_example_to_feature(data, args)
    for i, (sum, context) in enumerate(zip(sums, contexts)):
        sample = {'article': context,
                  'abstract': sum}
        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(sample, f, indent=4, ensure_ascii=False)
    with open(join(dump_dir), 'w') as f:
        for sum in sums:
            f.write(' '.join(sum) + '\n')

    print('finished in {}'.format(timedelta(seconds=time()-start)))

def main(args):
    for split in ['test.json']:  # evaluation of train data takes too long
        dump(split, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_mode', required=True, type=str)
    parser.add_argument('--turn_mode', required=True, type=str)
    parser.add_argument('--sum_mode', required=True, type=str)
    parser.add_argument('--context_mode', required=True, type=str)
    args = parser.parse_args()
    main(args)
