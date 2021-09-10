import sys
import re
import json

org_data_path = 'data/'
new_data_path = 'json_data/'

def convert(data, mode):
    new_data = []
    for sample in data:
        new_sample = {'session': [], 'summary': [], 'label': []}
        for turn in sample['Dialogue']:
            utter = {}
            if turn['speaker'] == 'Q':
                utter['type'] = '客户'
            else:
                utter['type'] = '客服'
            utter['word'] = turn['utterance'].split()
            utter['content'] = list(re.sub(' ', '', turn['utterance']))
            new_sample['session'].append(utter)
        if mode == 'final':
            new_sample['summary'] = list(''.join(sample['FinalSumm']))
            for qa in sample['QA']:
                new_sample['label'] += qa['QueSummUttIDs']
                new_sample['label'] += qa['AnsSummLongUttIDs']
        elif mode == 'user':
            new_sample['summary'] = list(''.join(sample['UserSumm']))
            for qa in sample['QA']:
                new_sample['label'] += qa['QueSummUttIDs']
        else:
            new_sample['summary'] = list(''.join(sample['AgentSumm']))
            for qa in sample['QA']:
                new_sample['label'] += qa['AnsSummShortUttIDs']
        new_sample['label'] = list(set(new_sample['label']))
        new_data.append(new_sample)
    return new_data


if __name__ == '__main__':
    #for mode in ['final', 'user', 'agent']:
    for mode in ['agent']:
        for name in ['train', 'val', 'test']:
            with open(org_data_path + name + '.json', 'r') as f:
                data = json.load(f)
            new_data = convert(data, mode)
            with open(new_data_path + mode + '/' + 'csds.' + name + '.json', 'w') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)