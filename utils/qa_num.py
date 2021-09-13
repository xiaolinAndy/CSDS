import json
import re
from metric import compute_rouge_n, compute_rouge_l

test_file = '../models/PGN/data/test.json'
model_name = ['PGN', 'fast_rl', 'bert_rl_mod']
#model_name = ['PGN']
#modes = ['final', 'user', 'agent']
modes = ['final']

THRESHOLD = 0.6

def get_ref_qa(mode):
    with open(test_file, 'r') as f:
        data = json.load(f)
    refs = []
    for sample in data:
        tmp_ref = []
        for qa in sample['QA']:
            if mode == 'final':
                tmp_str = qa['QueSumm'] + qa['AnsSummShort']
                tmp_ref.append(tmp_str)
            if mode == 'user':
                tmp_str = qa['QueSumm']
                tmp_ref.append(tmp_str)
            if mode == 'agent':
                tmp_str = qa['AnsSummLong']
                if tmp_str:
                    tmp_ref.append(tmp_str)
        refs.append(tmp_ref)
    return refs

def get_sents_qa_num(file_path, mode):
    sents = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            tmp_sum = []
            line = line.split('。')[:-1]
            #print(line)
            if mode == 'final':
                for i in range(int(len(line)/2)):
                    tmp_sum.append(line[2 * i] + '。' + line[2 * i + 1] + '。')
                if len(line) % 2 == 1:
                    tmp_sum.append(line[-1] + '。')
                sents.append(tmp_sum)
            else:
                sents.append([s + '。' for s in line])
    return sents

def catch(pred, ref):
    for p in pred:
        rouge_l = compute_rouge_l(p, ref, mode='f')
        if rouge_l > THRESHOLD:
            new_pred = pred.copy()
            new_pred.remove(p)
            return new_pred, True
        else:
            return pred, False

def cal_qa_acc(preds, refs):
    TP = 0.
    len_pred, len_ref = 0., 0.
    dict = {1: [0, 0, 0],
            2: [0, 0, 0],
            3: [0, 0, 0],
            4: [0, 0, 0],
            5: [0, 0, 0]}
    for pred, ref in zip(preds, refs):
        ref_ind = len(ref) if len(ref) < 5 else 5
        tmp_match = 0
        for qa in ref:
            if pred == []:
                break
            pred, match = catch(pred, qa)
            if match:
                TP += 1
                tmp_match += 1
                # print('True')
                # print(pred)
                # print(qa)
            # else:
            #     print('False')
            #     print(pred)
            #     print(qa)
        dict[ref_ind][0] += tmp_match
        dict[ref_ind][1] += len(ref)
        dict[ref_ind][2] += max(len(pred), tmp_match)
        len_pred += max(len(pred), tmp_match)
        len_ref += len(ref)
    prec = TP / len_pred
    recall = TP / len_ref
    f1 = 2 * prec * recall / (prec + recall)
    print(prec, recall, f1)
    # for k, v in dict.items():
    #     print(k, v[0]/v[1], v[0]/v[2])

if __name__ == '__main__':
    for mode in modes:
        for model in model_name:
            print(model, mode)
            pred_file = '../results/' + mode + '/' + model + '_true_preds.txt'
            preds = get_sents_qa_num(pred_file, mode)
            refs = get_ref_qa(mode)
            cal_qa_acc(preds, refs)


