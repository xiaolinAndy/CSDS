import re
import argparse
import files2rouge
import bert_score
import numpy as np
from moverscore_v2 import get_idf_dict, word_mover_score

final_dir = 'final_sum/'
user_dir = 'user_sum/'
agent_dir = 'agent_sum/'

def change_word2id(ref, pred):
    # if del_speaker:
    #     ref = re.sub('用户|客服|', '', ref)
    #     pred = re.sub('用户|客服|', '', pred)
    # ref = re.sub(' ', '', ref)
    # pred = re.sub(' ', '', pred)
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
    return ' '.join(ref_id), ' '.join(pred_id)

def get_sents(file_path):
    sents = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = re.sub(' ', '', line.strip())
            line = re.sub('<q>', '', line)
            sents.append(line)
    return sents

def get_data(name):
    data = []
    sents = get_sents(final_dir + name + '.txt')
    final_data.append(sents)
    user_files = ['PGN_both', 'PGN_only', 'fast_rl_both', 'fast_rl_only']
    for name in user_files:
        sents = get_sents(user_dir + name + '.txt')
        user_data.append(sents)
    agent_files = ['PGN_both', 'PGN_only', 'fast_rl_both', 'fast_rl_only']
    for name in agent_files:
        sents = get_sents(agent_dir + name + '.txt')
        agent_data.append(sents)
    return [final_data, user_data, agent_data]

def calculate(dir, files):
    refs = get_sents(dir + 'refs.txt')
    for name in files:
        preds = get_sents(dir + name + '.txt')
        # get rouge ids
        ref_ids, pred_ids = [], []
        for ref, pred in zip(refs, preds):
            ref_id, pred_id = change_word2id(ref, pred)
            ref_ids.append(ref_id)
            pred_ids.append(pred_id)
        with open(dir + 'ref_ids.txt', 'w') as f:
            for ref in ref_ids:
                f.write(ref + '\n')
        with open(dir + 'pred_ids.txt', 'w') as f:
            for pred in pred_ids:
                f.write(pred + '\n')
        print('Running rouge for ' + name + '-----------------------------')
        files2rouge.run(dir + 'pred_ids.txt', dir + 'ref_ids.txt')
        #
        # # run bertscore
        # prec, rec, f1 = bert_score.score(preds, refs, lang='zh')
        # prec = prec.mean().item()
        # rec = rec.mean().item()
        # f1 = f1.mean().item()
        # print('Running bert score for ' + name + '-----------------------------')
        # print('Precision: ', prec)
        # print('Recall: ', rec)
        # print('F1: ', f1)

        # run moverscore
        # idf_dict_hyp = get_idf_dict(preds)
        # idf_dict_ref = get_idf_dict(refs)
        # scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2)
        # avg_score = np.mean(np.array(scores))
        # print(avg_score)

def calculate_rouge(pred_txt, gold_txt):
    refs = get_sents(gold_txt)
    preds = get_sents(pred_txt)
    # get rouge ids
    ref_ids, pred_ids = [], []
    for ref, pred in zip(refs, preds):
        ref_id, pred_id = change_word2id(ref, pred)
        ref_ids.append(ref_id)
        pred_ids.append(pred_id)
    with open('logs/ref_ids.txt', 'w') as f:
        for ref in ref_ids:
            f.write(ref + '\n')
    with open('logs/pred_ids.txt', 'w') as f:
        for pred in pred_ids:
            f.write(pred + '\n')
    files2rouge.run('logs/pred_ids.txt', 'logs/ref_ids.txt')


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--pred_path', required=True)
    #parser.add_argument('--ref_path', required=True)
    #parser.add_argument('--del_speaker', type=bool, default=False)
    #args = parser.parse_args()
    #final_files = ['PGN', 'fast_rl_single', 'fast_rl_cont', 'human']
    #calculate(final_dir, final_files)
    # user_files = ['PGN_both', 'PGN_only', 'fast_rl_both', 'fast_rl_only', 'PGN_multi']
    # agent_files = ['PGN_both', 'PGN_only', 'fast_rl_both', 'fast_rl_only', 'PGN_multi']
    # user_files = ['PGN_multi']
    # agent_files = ['PGN_multi']
    # calculate(user_dir, user_files)
    # calculate(agent_dir, agent_files)
    #final_files = ['PGN', 'fast_rl_single', 'fast_rl_cont', 'human']
    #calculate(final_dir, final_files)
    pred_txt = 'logs/ext_bert.txt_step450.candidate'
    gold_txt = 'logs/ext_bert.txt_step450.gold'
    calculate_rouge(pred_txt, gold_txt)