# -*- coding:utf-8 -*-

import gc
import glob
import json
import os
import random
import torch
from os.path import join as pjoin

from collections import Counter
from rouge import Rouge
from others.logging import logger
from others.tokenization import BertTokenizer
from others.vocab_wrapper import VocabWrapper


def greedy_selection(doc, summ, summary_size):

    doc_sents = list(map(lambda x: x["original_txt"], doc))
    max_rouge = 0.0

    rouge = Rouge()
    selected = []
    while True:
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(doc_sents)):
            if (i in selected):
                continue
            c = selected + [i]
            temp_txt = " ".join([doc_sents[j] for j in c])
            if len(temp_txt.split()) > summary_size:
                continue
            rouge_score = rouge.get_scores(temp_txt, summ)
            rouge_1 = rouge_score[0]["rouge-1"]["r"]
            rouge_l = rouge_score[0]["rouge-l"]["r"]
            rouge_score = rouge_1 + rouge_l
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return selected


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.role_1 = '[unused3]'
        self.role_2 = '[unused4]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.unk_vid = self.tokenizer.vocab[self.unk_token]

    def preprocess_src(self, content, info=None):
        if_exceed_length = False

        if not (info == "客服" or info == '客户'):
            return None
        if len(content) < self.args.min_src_ntokens_per_sent:
            return None
        if len(content) > self.args.max_src_ntokens_per_sent:
            if_exceed_length = True

        original_txt = ' '.join(content)

        if self.args.truncated:
            content = content[:self.args.max_src_ntokens_per_sent]
        content_text = ' '.join(content).lower()
        content_subtokens = self.tokenizer.tokenize(content_text)

        # [CLS] + T0 + T1 + ... + Tn
        if info == '客服':
            src_subtokens = [self.cls_token, self.role_1] + content_subtokens
        else:
            src_subtokens = [self.cls_token, self.role_2] + content_subtokens
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        segments_ids = len(src_subtoken_idxs) * [0]

        return src_subtoken_idxs, segments_ids, original_txt, \
            src_subtokens, if_exceed_length

    def preprocess_summary(self, content):

        original_txt = ' '.join(content)

        content_text = ' '.join(content).lower()
        content_subtokens = self.tokenizer.tokenize(content_text)

        content_subtokens = [self.tgt_bos] + content_subtokens + [self.tgt_eos]
        subtoken_idxs = self.tokenizer.convert_tokens_to_ids(content_subtokens)

        return subtoken_idxs, original_txt, content_subtokens

    def integrate_dialogue(self, dialogue):
        src_tokens = [self.cls_token]
        segments_ids = [0]
        segment_id = 0
        for sent in dialogue:
            tokens = sent["src_tokens"][1:] + [self.sep_token]
            src_tokens.extend(tokens)
            segments_ids.extend([segment_id] * len(tokens))
            segment_id = 1 - segment_id
        src_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
        return {"src_id": src_ids, "segs": segments_ids}


def topic_info_generate(dialogue, file_counter):
    all_counter = Counter()
    customer_counter = Counter()
    agent_counter = Counter()

    for sent in dialogue:
        role = sent["role"]
        token_ids = sent["tokenized_id"]
        all_counter.update(token_ids)
        if role == "客服":
            agent_counter.update(token_ids)
        else:
            customer_counter.update(token_ids)
    file_counter['all'].update(all_counter.keys())
    file_counter['customer'].update(customer_counter.keys())
    file_counter['agent'].update(agent_counter.keys())
    file_counter['num'] += 1
    return {"all": all_counter, "customer": customer_counter, "agent": agent_counter}


def topic_summ_info_generate(dialogue, ex_labels):
    all_counter = Counter()
    customer_counter = Counter()
    agent_counter = Counter()

    for i, sent in enumerate(dialogue):
        if i in ex_labels:
            role = sent["role"]
            token_ids = sent["tokenized_id"]
            all_counter.update(token_ids)
            if role == "客服":
                agent_counter.update(token_ids)
            else:
                customer_counter.update(token_ids)
    return {"all": all_counter, "customer": customer_counter, "agent": agent_counter}


def format_to_bert(args, corpus_type=None):

    a_lst = []
    file_counter = {"all": Counter(), "customer": Counter(), "agent": Counter(), "num": 0, "voc_size": 0}
    if corpus_type is not None:
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, file_counter, pjoin(args.save_path, real_name.replace('json', 'pt'))))
    else:
        for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
            real_name = json_f.split('/')[-1]
            corpus_type = real_name.split('.')[1]
            a_lst.append((corpus_type, json_f, args, file_counter, pjoin(args.save_path, real_name.replace('json', 'pt'))))

    total_statistic = {
        "instances": 0,
        "total_turns": 0.,
        "processed_turns": 0.,
        "max_turns": -1,
        "turns_num": [0] * 11,
        "exceed_length_num": 0,
        "exceed_turns_num": 0,
        "total_src_length": 0.,
        "src_sent_length_num": [0] * 11,
        "src_token_length_num": [0] * 11,
        "total_tgt_length": 0
    }
    for d in a_lst:
        statistic = _format_to_bert(d)
        if statistic is None:
            continue
        total_statistic["instances"] += statistic["instances"]
        total_statistic["total_turns"] += statistic["total_turns"]
        total_statistic["processed_turns"] += statistic["processed_turns"]
        total_statistic["max_turns"] = max(total_statistic["max_turns"], statistic["max_turns"])
        total_statistic["exceed_length_num"] += statistic["exceed_length_num"]
        total_statistic["exceed_turns_num"] += statistic["exceed_turns_num"]
        total_statistic["total_src_length"] += statistic["total_src_length"]
        total_statistic["total_tgt_length"] += statistic["total_tgt_length"]
        for idx in range(len(total_statistic["turns_num"])):
            total_statistic["turns_num"][idx] += statistic["turns_num"][idx]
        for idx in range(len(total_statistic["src_sent_length_num"])):
            total_statistic["src_sent_length_num"][idx] += statistic["src_sent_length_num"][idx]
        for idx in range(len(total_statistic["src_token_length_num"])):
            total_statistic["src_token_length_num"][idx] += statistic["src_token_length_num"][idx]

    # save file counter
    save_file = pjoin(args.save_path, 'idf_info.pt')
    logger.info('Saving file counter to %s' % save_file)
    torch.save(file_counter, save_file)

    if total_statistic["instances"] > 0:
        logger.info("Total examples: %d" % total_statistic["instances"])
        logger.info("Average sentence number per dialogue: %f" % (total_statistic["total_turns"] / total_statistic["instances"]))
        logger.info("Processed average sentence number per dialogue: %f" % (total_statistic["processed_turns"] / total_statistic["instances"]))
        logger.info("Total sentences: %d" % total_statistic["total_turns"])
        logger.info("Processed sentences: %d" % total_statistic["processed_turns"])
        logger.info("Exceeded max sentence number dialogues: %d" % total_statistic["exceed_turns_num"])
        logger.info("Max dialogue sentences: %d" % total_statistic["max_turns"])
        for idx, num in enumerate(total_statistic["turns_num"]):
            logger.info("Dialogue sentences %d ~ %d: %d, %.2f%%" % (idx * 20, (idx+1) * 20, num, (num / total_statistic["instances"])))
        logger.info("Exceed length sentences number: %d" % total_statistic["exceed_length_num"])
        logger.info("Average src sentence length: %f" % (total_statistic["total_src_length"] / total_statistic["total_turns"]))
        for idx, num in enumerate(total_statistic["src_sent_length_num"]):
            logger.info("Sent length %d ~ %d: %d, %.2f%%" % (idx * 10, (idx+1) * 10, num, (num / total_statistic["total_turns"])))
        logger.info("Average src token length: %f" % (total_statistic["total_src_length"] / total_statistic["instances"]))
        for idx, num in enumerate(total_statistic["src_token_length_num"]):
            logger.info("token num %d ~ %d: %d, %.2f%%" % (idx * 300, (idx+1) * 300, num, (num / total_statistic["instances"])))
        logger.info("Average tgt length: %f" % (total_statistic["total_tgt_length"] / total_statistic["instances"]))


def _format_to_bert(params):
    _, json_file, args, file_counter, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    if args.tokenize:
        voc_wrapper = VocabWrapper(args.emb_mode)
        voc_wrapper.load_emb(args.emb_path)
        file_counter['voc_size'] = voc_wrapper.voc_size()

    datasets = []
    exceed_length_num = 0
    exceed_turns_num = 0
    total_src_length = 0.
    total_tgt_length = 0.
    src_length_sent_num = [0] * 11
    src_length_token_num = [0] * 11
    max_turns = 0
    turns_num = [0] * 11
    dialogue_turns = 0.
    processed_turns = 0.

    count = 0

    for dialogue in jobs:
        dialogue_b_data = []
        dialogue_token_num = 0
        for index, sent in enumerate(dialogue['session']):
            content = sent['content']
            role = sent['type']
            b_data = bert.preprocess_src(content, role)
            if (b_data is None):
                continue
            src_subtoken_idxs, segments_ids, original_txt, \
                src_subtokens, exceed_length = b_data
            b_data_dict = {"index": index, "src_id": src_subtoken_idxs,
                           "segs": segments_ids, "original_txt": original_txt,
                           "src_tokens": src_subtokens, "role": role}
            if args.tokenize:
                ids = map(lambda x: voc_wrapper.w2i(x), sent['word'])
                tokenized_id = [x for x in ids if x is not None]
                b_data_dict["tokenized_id"] = tokenized_id
            else:
                b_data_dict["tokenized_id"] = src_subtoken_idxs[2:]
            src_length_sent_num[min(len(src_subtoken_idxs) // 10, 10)] += 1
            dialogue_token_num += len(src_subtoken_idxs)
            total_src_length += len(src_subtoken_idxs)
            dialogue_b_data.append(b_data_dict)
            if exceed_length:
                exceed_length_num += 1
            if len(dialogue_b_data) >= args.max_turns:
                exceed_turns_num += 1
                if args.truncated:
                    break
        dialogue_example = {"session": dialogue_b_data}
        dialogue_integrated = bert.integrate_dialogue(dialogue_b_data)
        topic_info = topic_info_generate(dialogue_b_data, file_counter)
        dialogue_example["dialogue"] = dialogue_integrated
        dialogue_example["topic_info"] = topic_info
        # test & dev data process
        if "summary" in dialogue.keys():
            content = dialogue["summary"]
            summ_b_data = bert.preprocess_summary(content)
            subtoken_idxs, original_txt, content_subtokens = summ_b_data
            total_tgt_length += len(subtoken_idxs)
            b_data_dict = {"id": subtoken_idxs,
                           "original_txt": original_txt,
                           "content_tokens": content_subtokens}
            if args.add_ex_label:
                #ex_labels = greedy_selection(dialogue_b_data, original_txt, args.ex_max_token_num)
                ex_labels = dialogue['label']
                topic_summ_info = topic_summ_info_generate(dialogue_b_data, ex_labels)
                b_data_dict["ex_labels"] = ex_labels
                b_data_dict["topic_summ_info"] = topic_summ_info
            dialogue_example["summary"] = b_data_dict

        if len(dialogue_b_data) >= args.min_turns:
            datasets.append(dialogue_example)
            turns_num[min(len(dialogue_b_data) // 20, 10)] += 1
            src_length_token_num[min(dialogue_token_num // 300, 10)] += 1
            max_turns = max(max_turns, len(dialogue_b_data))
            dialogue_turns += len(dialogue['session'])
            processed_turns += len(dialogue_b_data)

            count += 1
            if count % 50 == 0:
                print(count)

    statistic = {
        "instances": len(datasets),
        "total_turns": dialogue_turns,
        "processed_turns": processed_turns,
        "max_turns": max_turns,
        "turns_num": turns_num,
        "exceed_length_num": exceed_length_num,
        "exceed_turns_num": exceed_turns_num,
        "total_src_length": total_src_length,
        "src_sent_length_num": src_length_sent_num,
        "src_token_length_num": src_length_token_num,
        "total_tgt_length": total_tgt_length
    }

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    return statistic
