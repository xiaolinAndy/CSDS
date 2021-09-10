import re
import files2rouge

def get_sents_str(f):
    sents = []
    for line in f.readlines():
        line = re.sub(' ', '', line.strip())
        sents.append(line)
    return sents

def change_word2id(ref, pred):
    ref = re.sub(' ', '', ref)
    pred = re.sub(' ', '', pred)
    pred = re.sub('<q>', '', pred)
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

def cal_rouge(pred_name, ref_name):
    refs = get_sents_str(ref_name)
    preds = get_sents_str(pred_name)
    # write ids
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

def cal_rouge_path(pred_name, ref_name):
    with open(pred_name, 'r') as f:
        refs = get_sents_str(f)
    with open(ref_name, 'r') as f:
        preds = get_sents_str(f)
    # write ids
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