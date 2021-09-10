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


def get_extract_label(art_sents, abs_sents, labels):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    if len(labels) != len(abs_sents):
        print(labels, abs_sents)
        exit()
    for abst, label in zip(abs_sents, labels):
        if len(label) == 1:
            extracted.append(label[0])
            scores.append(1)
        else:
            max_rouge = 0
            max_ind = -1
            for i in label:
                rouge = compute_rouge_l(output=art_sents[i], reference=abst, mode='r')
                if max_rouge < rouge:
                    max_rouge = rouge
                    max_ind = i
            assert max_ind != -1
            extracted.append(max_ind)
            scores.append(max_rouge)
    return extracted, scores

def get_new_label(label_map, sample, sum_mode):
    new_labels = []
    for qa in sample['QA']:
        tmp_ids = []
        for id in qa['QueSummUttIDs']:
            tmp_ids.append(label_map[id])
        tmp_ids = list(set(tmp_ids))
        if sum_mode == 'final' or sum_mode == 'user':
            new_labels.append(tmp_ids)
        if qa['AnsSummLongUttIDs']:
            tmp_ids = []
            for id in qa['AnsSummLongUttIDs']:
                tmp_ids.append(label_map[id])
            tmp_ids = list(set(tmp_ids))
            if sum_mode == 'agent':
                new_labels.append(tmp_ids)
        if qa['AnsSummShortUttIDs']:
            tmp_ids = []
            for id in qa['AnsSummShortUttIDs']:
                tmp_ids.append(label_map[id])
            tmp_ids = list(set(tmp_ids))
            if sum_mode == 'final':
                new_labels.append(tmp_ids)
    return new_labels


def convert_example_to_feature(data, args):
    sums, contexts, turns, labels = [], [], [], []
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
            split_sum = []
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
                if args.complete:
                    sent = jieba.lcut(''.join(turn['new_utterance'].split()))
                else:
                    sent = turn['utterance'].split()
                for word in sent:
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
            label_map = {}
            for i, turn in enumerate(sample['Dialogue']):
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
                    if args.complete:
                        sent = jieba.lcut(''.join(turn['new_utterance'].split()))
                    else:
                        sent = turn['utterance'].split()
                    for word in sent:
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
                label_map[i] = len(context)
            if args.context_mode == 'both':
                context.append(' '.join(tmp_utt))
            elif args.context_mode == 'user' and last_speaker == 'Q':
                context.append(' '.join(tmp_utt))
            elif args.context_mode == 'agent' and last_speaker == 'A':
                context.append(' '.join(tmp_utt))
            mod_label = get_new_label(label_map, sample, args.sum_mode)
            labels.append(mod_label)
        contexts.append(context)
    return sums, contexts, labels

def label(split, data_file, args):
    start = time()
    print('start processing {} split...'.format(split))
    data_path = join(DATA_DIR, data_file)
    data_dir = join(DATA_DIR, split) + '/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(data_path, 'r') as f:
        data = json.load(f)
    sums, contexts, labels = convert_example_to_feature(data, args)
    for i, (sum, context, label) in enumerate(zip(sums, contexts, labels)):
        #extracted, scores = get_extract_label(context, sum)
        extracted, scores = get_extract_label(context, sum, label)
        sample = {'article': context,
                  'abstract': sum}
        sample['extracted'] = extracted
        sample['score'] = scores
        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(sample, f, indent=4, ensure_ascii=False)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(args):
    train_file = 'complete_train.json' if args.complete else 'train.json'
    val_file = 'complete_val.json' if args.complete else 'val.json'
    splits = ['val', 'train']
    for split, name in zip(splits, [val_file, train_file]):  # no need of extraction label when testing
        label(split, name, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_mode', required=True, type=str)
    parser.add_argument('--turn_mode', required=True, type=str)
    parser.add_argument('--sum_mode', required=True, type=str)
    parser.add_argument('--context_mode', required=True, type=str)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--complete", action='store_true')
    args = parser.parse_args()
    main(args)
