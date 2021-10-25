import re
import os
import json
import files2rouge
import bert_score
import numpy as np
import scipy.stats
from moverscore_v2 import get_idf_dict, word_mover_score
from nltk.translate.bleu_score import corpus_bleu

model_name = ['bert_ext', 'PGN', 'fast_rl', 'fast_rl_mod', 'bert_abs_block5', 'bert_rl_mod_block5']
#modes = ['final', 'user', 'agent']
modes = ['final']

def get_sents_str(file_path):
    sents = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            sents.append(line)
    return sents

def change_word2id_split(ref, pred):
    ref_id, pred_id = [], []
    tmp_dict = {'%': 0}
    new_index = 1
    words = list(ref)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            ref_id.append(str(new_index))
            new_index += 1
        else:
            ref_id.append(str(tmp_dict[w]))
        if w == '。':
            ref_id.append(str(0))
    words = list(pred)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            pred_id.append(str(new_index))
            new_index += 1
        else:
            pred_id.append(str(tmp_dict[w]))
        if w == '。':
            pred_id.append(str(0))
    return ' '.join(ref_id), ' '.join(pred_id)

def run_rouge(pred_file, ref_file):
    refs = get_sents_str(ref_file)
    preds = get_sents_str(pred_file)
    # write ids
    ref_ids, pred_ids = [], []
    for ref, pred in zip(refs, preds):
        ref_id, pred_id = change_word2id(ref, pred)
        ref_ids.append(ref_id)
        pred_ids.append(pred_id)
    with open('../tmp/ref_ids.txt', 'w') as f:
        for ref in ref_ids:
            f.write(ref + '\n')
    with open('../tmp/pred_ids.txt', 'w') as f:
        for pred in pred_ids:
            f.write(pred + '\n')
    files2rouge.run('../tmp/pred_ids.txt', '../tmp/ref_ids.txt')

def read_rouge_score(name):
    with open(name, 'r') as f:
        lines = f.readlines()
    r1 = lines[3][21:28]
    r2 = lines[7][21:28]
    rl = lines[11][21:28]
    return [float(r1), float(r2), float(rl)]

def get_js(pred, ref):
    bi_grams = {}
    for i in range(len(ref) - 1):
        if ref[i] + ref[i + 1] not in bi_grams:
            bi_grams[ref[i] + ref[i + 1]] = len(bi_grams)
    for i in range(len(pred) - 1):
        if pred[i] + pred[i + 1] not in bi_grams:
            bi_grams[pred[i] + pred[i + 1]] = len(bi_grams)
    p = np.zeros(len(bi_grams))
    q = np.zeros(len(bi_grams))
    for i in range(len(ref) - 1):
        q[bi_grams[ref[i] + ref[i + 1]]] += 1
    q = q / np.sum(q)
    for i in range(len(pred) - 1):
        p[bi_grams[pred[i] + pred[i + 1]]] += 1
    p = p / np.sum(p)
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

def calculate(pred_file, ref_file, mode, model):
    print('Mode: ', mode, ' Model: ', model, '-----------------')
    refs = get_sents_str(ref_file)
    preds = get_sents_str(pred_file)
    #get rouge ids
    ref_ids, pred_ids = [], []
    for ref, pred in zip(refs, preds):
        ref_id, pred_id = change_word2id_split(ref, pred)
        ref_ids.append(ref_id)
        pred_ids.append(pred_id)
    with open('ref_ids.txt', 'w') as f:
        for ref in ref_ids:
            f.write(ref + '\n')
    with open('pred_ids.txt', 'w') as f:
        for pred in pred_ids:
            f.write(pred + '\n')
    print('Running rouge for ' + mode + ' ' + model + '-----------------------------')
    os.system('files2rouge ref_ids.txt pred_ids.txt -s rouge.txt -e 0')
    rouge_scores = read_rouge_score('rouge.txt')

    #run bleu
    bleu_preds = [list(s) for s in preds]
    bleu_refs = [[list(s)] for s in refs]
    bleu_score = corpus_bleu(bleu_refs, bleu_preds)
    print('Running BLEU for ' + mode + ' ' + model + '-----------------------------')
    print('BLEU: ', bleu_score)

    # run bertscore
    prec, rec, f1 = bert_score.score(preds, refs, lang='zh')
    prec = prec.mean().item()
    rec = rec.mean().item()
    bertscore_f1 = f1.mean().item()
    print('Running BERTScore for ' + mode + ' ' + model + '-----------------------------')
    print('Precision: ', prec)
    print('Recall: ', rec)
    print('F1: ', bertscore_f1)

    # run moverscore
    print('Running MoverScore for ' + mode + ' ' + model + '-----------------------------')
    idf_dict_hyp = get_idf_dict(preds)
    idf_dict_ref = get_idf_dict(refs)
    scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2, batch_size=16)
    mover_avg_score = np.mean(np.array(scores))
    print('F1: ', mover_avg_score)

    auto_scores = rouge_scores + [bleu_score, bertscore_f1, mover_avg_score]
    return auto_scores





if __name__ == '__main__':
    auto_metric_results = {}
    for mode in modes:
        auto_metric_results[mode] = {}
        for model in model_name:
            pred_file = '../results/' + mode + '/' + model + '_true_preds.txt'
            if 'bert_rl' in model:
                ref_file = '../results/' + mode + '/' + model + '_true_refs.txt'
            else:
                ref_file = '../results/' + mode + '/gold_refs.txt'
            auto_scores = calculate(pred_file, ref_file, mode, model)
            auto_metric_results[mode][model] = auto_scores
    with open('../results/tmp_auto_metric_results.json', 'w') as f:
        json.dump(auto_metric_results, f, indent=4)
