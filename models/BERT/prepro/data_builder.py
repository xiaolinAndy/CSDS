import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
from pathlib import Path

import torch
from multiprocessing import Pool
from tqdm import tqdm

from others.logging import logger
from others.tokenization import BertTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET

import jieba

import re


nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]

def cut_paragraph(paragraph):
  splited = [] 
  separator = list('。；？！.;!?')
  sen = []
  is_divisible = True
  for char in paragraph:
    sen.append(char)
    if char == '“':
      is_divisible = False
    if char == '”':
      is_divisible = True
      
    if char in separator and is_divisible:
      splited.append(''.join(sen))
      sen = []
    
  if len(sen) != 0:
    splited.append(''.join(sen))

  return splited


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)



def load_json(p):
    with p.open('r', encoding = 'utf-8') as f:
        json_data = json.load(f)
        article = json_data["article"]
        abstract = json_data["abstract"]
        source = [[tk.lower() for tk in sen.strip().split()] for sen in article]
        tgt = [[tk.lower() for tk in sen.strip().split()] for sen in abstract]

        source = [clean(' '.join(sent)).split() for sent in source]
        tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt



def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, pretrained_path, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.eou = '[unused98]'
        self.tgt_bos = '[unused99]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    # def preprocess(self, src, tgt, sent_labels, is_dialogue, use_bert_basic_tokenizer, is_test=False):
    #
    #     if ((not is_test) and len(src) == 0):
    #         return None
    #
    #     original_src_txt = [' '.join(s) for s in src]
    #
    #     idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
    #
    #     _sent_labels = [0] * len(src)
    #     for l in sent_labels:
    #         try:
    #             _sent_labels[l] = 1
    #         except:
    #             print(l)
    #             print(len(src))
    #             print(src)
    #             exit()
    #
    #     src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
    #     sent_labels = [_sent_labels[i] for i in idxs]
    #     src = src[:self.args.max_src_nsents]
    #     sent_labels = sent_labels[:self.args.max_src_nsents]
    #
    #     if ((not is_test) and len(src) < self.args.min_src_nsents):
    #         return None
    #
    #     src_txt = [' '.join(sent) for sent in src]
    #     if is_dialogue:
    #         text = ' {} {} {} '.format(self.eou, self.sep_token, self.cls_token).join(src_txt)
    #     else:
    #         text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
    #
    #     src_subtokens = self.tokenizer.tokenize(text)
    #     if is_dialogue:
    #         src_subtokens = [self.cls_token] + src_subtokens + [self.eou] + [self.sep_token]
    #     else:
    #         src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
    #     src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
    #     _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
    #     segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
    #     segments_ids = []
    #     for i, s in enumerate(segs):
    #         if (i % 2 == 0):
    #             segments_ids += s * [0]
    #         else:
    #             segments_ids += s * [1]
    #     cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
    #     sent_labels = sent_labels[:len(cls_ids)]
    #
    #     tgt_subtokens_str = '[unused99] ' + ' [unused2] '.join(
    #         [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
    #     tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
    #     if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
    #         return None
    #
    #     tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
    #
    #     tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
    #     src_txt = [original_src_txt[i] for i in idxs]
    #
    #     return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

    def preprocess(self, src, tgt, sent_labels, is_dialogue, use_bert_basic_tokenizer, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            try:
                _sent_labels[l] = 1
            except:
                print(l)
                print(len(src))
                print(src)
                exit()

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        if is_dialogue:
            text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        else:
            text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        #print(text)
        #text = re.sub(' ', '', text)
        #print(text)
        src_subtokens = self.tokenizer.tokenize(text, use_bert_basic_tokenizer=use_bert_basic_tokenizer)
        if is_dialogue:
            src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        else:
            src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused99] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        # print(src_subtokens, tgt_subtoken, tgt_subtoken_idxs)
        # exit()

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']

    read_root_path = Path(args.raw_path)
    save_root_path = Path(args.save_path)
    for corpus_type in datasets:
        save_path = save_root_path / corpus_type
        save_path.mkdir(parents=True, exist_ok=True)
        a_lst = []
        for fp in (read_root_path / corpus_type).iterdir():
            a_lst.append((corpus_type, fp, args, save_path / f'{fp.stem}.bert.pt'))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, fp, args, save_file = params
    is_test = corpus_type == 'test'
    if (save_file.exists()):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info(f'Processing {fp.stem}' )
    jobs = json.load(fp.open('r', encoding = 'utf-8'))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        b_data = bert.preprocess(source, tgt, sent_labels, args.is_dialogue, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                        "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                        'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpora = {'train': [], 'valid': [], 'test': []}
    
    read_root_path = Path(args.raw_path)
    for corpus_type in ['valid', 'test', 'train']:
        read_path = read_root_path / corpus_type
        for fp in read_path.iterdir():
            corpora[corpus_type].append(fp)
    
    save_root_path = Path(args.save_path)
    for corpus_type in ['train', 'valid', 'test']:
        save_path = save_root_path / corpus_type
        save_path.mkdir(parents=True, exist_ok=True)
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in tqdm(pool.imap_unordered(_format_to_lines, a_lst)):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                with (save_path / f'{p_ct}.json').open('w', encoding='utf-8') as s_f:
                    s_f.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            with (save_path / f'{p_ct}.json').open('w', encoding='utf-8') as s_f:
                # save.write('\n'.join(dataset))
                s_f.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    source, tgt = load_json(f)
    return {'src': source, 'tgt': tgt}


def nlpcc_format_to_lines(args):
    corpora = {'train': [], 'valid': [], 'test': []}
    
    read_root_path = Path(args.raw_path)
    save_root_path = Path(args.save_path)
    for corpus_type in ['valid', 'test', 'train']:
        read_path = read_root_path / f'{corpus_type}.txt'
        save_path = save_root_path / f'{corpus_type}.json'
        with read_path.open('r', encoding = 'utf-8') as r_f:
            for line in r_f:
                json_data = json.loads(line)
                summ = [list(jieba.cut(sen)) for sen in cut_paragraph(json_data["summarization"])]
                arti = [list(jieba.cut(sen)) for para in json_data["article"].split('<Paragraph>')\
                                                for sen in cut_paragraph(para)]
                corpora[corpus_type].append({'src': arti, 'tgt': summ})
        with save_path.open('w', encoding='utf-8') as w_f:
            w_f.write(json.dumps(corpora[corpus_type]))

def convert_bio_label(bio_indexes, total_len):
    labels = [0] * total_len
    for (start, end) in bio_indexes:
        labels[start] = 1
        for i in range(start + 1, end + 1):
            labels[i] = 2
    return labels

def DS_format_to_lines(context_mode, summ_mode, args):
    assert summ_mode in ['final', 'user', 'agent']
    assert context_mode in ['both', 'user', 'agent']
    corpora = {'train': [], 'val': [], 'test':[]}

    read_root_path = Path(args.raw_path)
    save_root_path = Path(args.save_path) / f'{context_mode}' / f'{summ_mode}'
    save_root_path.mkdir(exist_ok=True, parents=True) 
    for corpus_type in ['val', 'test', 'train']:
        read_path = read_root_path / f'{corpus_type}.json'
        save_path = save_root_path / f'{corpus_type}.json'
        
        with read_path.open('r', encoding = 'utf-8') as r_f:
            json_data = json.load(r_f)
            for sample in json_data:     
                if summ_mode == 'final':
                    summ = [list(jieba.cut(sen)) for sen in sample['FinalSumm']] # list
                elif summ_mode == 'user':
                    summ = [list(jieba.cut(sen)) for sen in sample['UserSumm']]
                elif summ_mode == 'agent':
                    summ = [list(jieba.cut(sen)) for sen in sample['AgentSumm']]

                ext_label = []
                bio_indexs = []
                for qa in sample["QA"]:
                    if qa != []:
                        if summ_mode == 'final':
                            ext_label = ext_label + qa["QueSummUttIDs"] + qa["AnsSummLongUttIDs"] # list
                            start = min(qa["QueSummUttIDs"] + qa["AnsSummLongUttIDs"])
                            end = max(qa["QueSummUttIDs"] + qa["AnsSummLongUttIDs"])
                            bio_indexs.append([start, end])
                        elif summ_mode == 'user':
                            ext_label = ext_label + qa["QueSummUttIDs"]
                        elif summ_mode == 'agent':
                            ext_label = ext_label + qa["AnsSummLongUttIDs"]
                
                context = []
                for turn in sample['Dialogue']:
                    tmp_utt = []
                    if args.add_prefix:
                        if turn['speaker'] == 'Q':
                            tmp_utt += [sample['QRole'], ':']
                        else:
                            tmp_utt += ['客服', ':']
                    for word in turn['utterance'].split():
                        if len(word) > 2 and word[0] == '[' and word[-1] == ']':
                            tmp_utt += ['[', word[1:-1], ']']
                        else:
                            tmp_utt.append(word)
                    # tmp_utt = ' '.join(tmp_utt)
                    if context_mode == 'both':
                        context.append(tmp_utt)
                    elif context_mode == 'user' and turn['speaker'] == 'Q':
                        context.append(tmp_utt)
                    elif context_mode == 'agent' and turn['speaker'] == 'A':
                        context.append(tmp_utt)
                bio_label = convert_bio_label(bio_indexs, len(context))
                
                corpora[corpus_type].append({'src': context, 'tgt': summ, 'ext':ext_label, 'bio': bio_label})
                
        with save_path.open('w', encoding='utf-8') as w_f:
            w_f.write(json.dumps(corpora[corpus_type], indent=4, ensure_ascii=False))

def nlpcc_format_to_bert(pretrained_path, args):
    corpora = {'train': [], 'valid': [], 'test': []}
    bert = BertData(pretrained_path, args)
    read_root_path = Path(args.raw_path)
    save_root_path = Path(args.save_path)
    for corpus_type in corpora:
        is_test = corpus_type == 'test'

        read_path = read_root_path / f'{corpus_type}.json'
        save_path = save_root_path / f'{corpus_type}.bert.bin'

        logger.info(f'Processing {read_path.stem}' )
        jobs = json.load(read_path.open('r', encoding = 'utf-8'))
        
        for d in tqdm(jobs):
            source, tgt = d['src'], d['tgt']

            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
            b_data = bert.preprocess(source, tgt, sent_labels, args.is_dialogue, use_bert_basic_tokenizer=True, is_test=is_test)

            if (b_data is None):
                continue
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                            "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                            'src_txt': src_txt, "tgt_txt": tgt_txt}
            
            corpora[corpus_type].append(b_data_dict)

        logger.info('Processed instances %d' % len(corpora[corpus_type]))
        logger.info('Saving to %s' % save_path)
        torch.save(corpora[corpus_type], save_path)


def DS_format_to_bert(pretrained_path, context_mode, summ_mode, args):
    corpora = {'train': [], 'val': [], 'test': []}
    bert = BertData(pretrained_path, args)
    read_root_path = Path(args.raw_path) / f'{context_mode}' / f'{summ_mode}' 
    

    for corpus_type in corpora:
        save_root_path = Path(args.save_path) / f'{context_mode}' / f'{summ_mode}' / corpus_type
        save_root_path.mkdir(exist_ok=True, parents=True) 

        is_test = corpus_type[:4] == 'test'

        read_path = read_root_path / f'{corpus_type}.json'
        save_path = save_root_path / f'{corpus_type}.bert.bin'

        logger.info(f'Processing {read_path.stem}' )
        jobs = json.load(read_path.open('r', encoding = 'utf-8'))
        
        for d in tqdm(jobs):
            source, tgt, sent_labels = d['src'], d['tgt'], d['ext']
            # sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
            b_data = bert.preprocess(source, tgt, sent_labels, args.is_dialogue, use_bert_basic_tokenizer=True, is_test=is_test)

            if (b_data is None):
                continue
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
            # change extractive labels
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                            "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                            'src_txt': src_txt, "tgt_txt": tgt_txt}
            
            corpora[corpus_type].append(b_data_dict)
            # print(sent_labels)
            # exit()

        logger.info('Processed instances %d' % len(corpora[corpus_type]))
        logger.info('Saving to %s' % save_path)
        torch.save(corpora[corpus_type], save_path)
        

def _format_to_bert(params):
    corpus_type, fp, args, save_file = params
    is_test = corpus_type == 'test'
    if (save_file.exists()):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info(f'Processing {fp.stem}' )
    jobs = json.load(fp.open('r', encoding = 'utf-8'))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        b_data = bert.preprocess(source, tgt, sent_labels, args.is_dialogue, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                        "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                        'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None
