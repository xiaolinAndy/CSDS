from pathlib import Path
import numpy as np
import json
from collections import Counter
import jieba
from tqdm import tqdm


def get_nlpcc_vocab(nlpcc_path, save_path, granularity):
  nlpcc_vocab_counter = Counter()
  for file_path in nlpcc_path.iterdir():
    with file_path.open('r', encoding='utf-8') as r_f:
      for line in tqdm(r_f):
        json_data = json.loads(line)
        summ = json_data['summarization']
        article = ''.join(json_data['article'].split('<Paragraph>'))
        if granularity == 'word':
          nlpcc_vocab_counter.update(list(jieba.cut(summ+article)))
        elif granularity == 'char':
          nlpcc_vocab_counter.update(summ+article)
  dialogue_vocab = [w for w, _ in nlpcc_vocab_counter.most_common()]
  print(f'nlpcc {granularity} vocab size: {len(dialogue_vocab)}')

  with (save_path / f'nlpcc_vocab_{granularity}').open('w', encoding='utf-8') as w_f:
    for char in tqdm(dialogue_vocab):
      w_f.write(char + '\n')

  return

def get_dialogue_vocab(sums, contexts, save_path, granularity):
  dialogue_vocab_counter = Counter()
  for sum in sums:
    if granularity == 'word':
      dialogue_vocab_counter.update(jieba.lcut(sum))
  for context in contexts:
    for utt in context:
      if granularity == 'word':
        dialogue_vocab_counter.update(utt.split())
  dialogue_vocab = [w for w, num in dialogue_vocab_counter.most_common()]
  print(f'dialogue {granularity} vocab num: {len(dialogue_vocab)}')

  with (save_path / f'dialogue_vocab_{granularity}').open('w', encoding='utf-8') as w_f:
    for w in tqdm(dialogue_vocab):
      w_f.write(w + '\n')

  return


def get_tencent_vocab(tencent_path, save_path):
  tencent_vocab = []
  with tencent_path.open('r', encoding = 'utf-8') as r_f:
    head = r_f.readline().strip()
    print(f'tencent embedding info: {head}')
    for line in tqdm(r_f):
      split_line = line.split()
      word = ' '.join(split_line[0:-200])
      tencent_vocab.append(word)
  print(f'tencent vocab size: {len(tencent_vocab)}')

  with (save_path / 'tencent_vocab').open('w', encoding='utf-8') as w_f:
    for word in tqdm(tencent_vocab):
      w_f.write(word + '\n')

  return


def load_vocab(vocab_path):
  with vocab_path.open('r', encoding='utf-8') as r_f:
    vocab = [word.strip() for word in tqdm(r_f)]
  return vocab


def operate_vocab(vocab_root_path, vocab_a_name, vocab_b_name, operator):
  assert operator in ['intersect', 'sub']
  vocab_a = load_vocab(vocab_root_path / vocab_a_name)
  vocab_b = load_vocab(vocab_root_path / vocab_b_name)

  vocab_a_set = set(vocab_a)
  vocab_b_set = set(vocab_b)

  print(f'{vocab_a_name}: {len(vocab_a)}\n{vocab_b_name}: {len(vocab_b)}')
  print(f'{vocab_a_name} set: {len(vocab_a_set)}\n{vocab_b_name} set: {len(vocab_b_set)}')
  
  if operator == 'intersect':
    result = vocab_a_set & vocab_b_set
  elif operator == 'sub':
    result = vocab_a_set - vocab_b_set
  
  print(f'{operator} of {vocab_a_name} {vocab_b_name} size: {len(result)}')
  with (vocab_root_path / f'{operator}_{vocab_a_name.split("_")[0]}_{vocab_b_name.split("_")[0]}_{vocab_a_name.split("_")[-1]}').open('w', encoding='utf-8') as w_f:
    for word in tqdm(result):
      w_f.write(word+'\n')

    
def get_tencent_embedding(vocab, tencent_embedding_path, save_path):
  word2embed = {}
  with tencent_embedding_path.open('r', encoding = 'utf-8') as r_f:
    head = r_f.readline().strip()
    print(f'tencent embedding info: {head}')
    for line in tqdm(r_f):
      split_line = line.split()
      word = ' '.join(split_line[0:-200])
      if word in vocab:
        embed = split_line[-200:]
        word2embed[word] = embed
  
  with save_path.open('w', encoding='utf-8') as w_f:
    # for word, embed in tqdm(word2embed.items()):
    #   w_f.write(word + ' ' + ' '.join(embed) + '\n')
    for word in vocab:
      if word in word2embed.keys():
        w_f.write(word + ' ' + ' '.join(word2embed[word]) + '\n')
      else:
        embed = ['0'] * 200
        w_f.write(word + ' ' + ' '.join(embed) + '\n')



class Tokenizer(object):
  def __init__(self, embed_path, embed_dim, max_vocab_size):
    self.embed_path = Path(embed_path)
    self.embed_dim = embed_dim
    self.auxiliary_token_dict = {"<PAD>":0, "<UNK>":1, "<START>":2, "<END>":3, "<EOU>":4, "<BOP>": 5, "<EOP>": 6}
    self.max_vocab_size = max_vocab_size
    self.vocab_size, self.token2idx_dict, self.embedding = self._get_vocab()
    self.idx2token_dict = {idx:token for token, idx in self.token2idx_dict.items()}


  def _get_vocab(self):
    print('loading embedding...')
    token2idx_dict = self.auxiliary_token_dict
    token_id = len(self.auxiliary_token_dict)
    embedding = [[0.0]*self.embed_dim]*len(self.auxiliary_token_dict)
    
    with self.embed_path.open('r', encoding='utf-8') as r_f:
      for line in r_f:
        split_line = line.split()
        token = split_line[:-self.embed_dim]
        if len(token) != 0:
          token = token[0]
          embed = [float(num) for num in split_line[-self.embed_dim:]]
          token2idx_dict[token] = token_id
          embedding.append(embed)
          token_id += 1
          if token_id >= self.max_vocab_size:
            break

    print(f'vocab size: {token_id}')
    return token_id, token2idx_dict, np.array(embedding)

  def token2idx(self, token):
    return self.token2idx_dict.get(token, self.token2idx_dict['<UNK>'])

  def idx2token(self, idx):
    if idx not in self.idx2token_dict:
      #assert idx in self.idx2token_dict, f'token id {idx} is out of range'
      raise ValueError('Id not found in vocab: %d' % idx)
    return self.idx2token_dict[idx]

  def tokenize(self, sen, is_splited):
    if not is_splited:
      sen = list(jieba.cut(sen))
    else:
      sen = sen.split()
    return list(map(self.token2idx, sen))

  def split(self, sen):
    sen = list(jieba.cut(sen))
    return sen

  def untokenize(self, ids):
    return list(map(self.idx2token, ids))

  def article2ids(self, article_words):
    ids = []
    oovs = []
    unk_id = self.token2idx('<UNK>')
    for w in article_words:
      i = self.token2idx(w)
      if i == unk_id:  # If w is OOV
        if w not in oovs:  # Add to list of OOVs
          oovs.append(w)
        oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
        ids.append(self.vocab_size + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
      else:
        ids.append(i)
    return ids, oovs

  def abstract2ids(self, abstract_words, article_oovs):
    ids = []
    unk_id = self.token2idx('<UNK>')
    for w in abstract_words:
      i = self.token2idx(w)
      if i == unk_id:  # If w is an OOV word
        if w in article_oovs:  # If w is an in-article OOV
          vocab_idx = self.vocab_size  + article_oovs.index(w)  # Map to its temporary article OOV number
          ids.append(vocab_idx)
        else:  # If w is an out-of-article OOV
          ids.append(unk_id)  # Map to the UNK token id
      else:
        ids.append(i)
    return ids

  def outputids2words(self, id_list, article_oovs):
    words = []
    for i in id_list:
      try:
        w = self.idx2token(i)  # might be [UNK]
      except ValueError as e:  # w is OOV
        assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
        article_oov_idx = i - self.vocab_size
        try:
          w = article_oovs[article_oov_idx]
        except ValueError as e:  # i doesn't correspond to an article oov
          raise ValueError(
            'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
            i, article_oov_idx, len(article_oovs)))
      words.append(w)
    return words


