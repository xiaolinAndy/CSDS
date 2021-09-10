import gc
import glob
import random
import torch
import math

from collections import Counter
from others.logging import logger
from torchtext.vocab import Vocab


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, args, data=None, device=None, is_test=False, idf_info=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.args = args
            self.batch_size = len(data)
            session_src = (x[2] for x in data)
            ex_segs = [len(s) for s in session_src]

            if args.src_data_mode == 'utt':
                src = torch.tensor(self._pad(sum((x[2] for x in data), []), 0))
                segs = torch.tensor(self._pad(sum((x[3] for x in data), []), 0))
            else:
                src = torch.tensor(self._pad([x[0] for x in data], 0))
                segs = torch.tensor(self._pad([x[1] for x in data], 0))
            tgt = torch.tensor(self._pad([x[5] for x in data], 0))
            mask_src = ~(src == 0)
            mask_tgt = ~(tgt == 0)

            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))
            setattr(self, 'ex_segs', ex_segs)

            original_str = sum((x[4] for x in data), [])
            setattr(self, 'original_str', original_str)
            tgt_txt = [x[6] for x in data]
            setattr(self, 'tgt_txt', tgt_txt)
            tgt_labels = [x[7] for x in data]
            setattr(self, 'tgt_labels', tgt_labels)

            if args.topic_model:
                all_bow, customer_bow, agent_bow = self.generate_bow(data, idf_info)
                setattr(self, 'all_bow', all_bow.to(device))
                setattr(self, 'customer_bow', customer_bow.to(device))
                setattr(self, 'agent_bow', agent_bow.to(device))

                summ_all_bow, summ_cust_bow, summ_agent_bow = self.generate_summ_bow(data, idf_info)
                setattr(self, 'summ_all', summ_all_bow.to(device))
                setattr(self, 'summ_customer', summ_cust_bow.to(device))
                setattr(self, 'summ_agent', summ_agent_bow.to(device))

            if args.copy_attn:
                src_map = self.make_src_map([x[-3] for x in data])
                align = torch.tensor(self._pad([x[-2] for x in data], 0))

                setattr(self, 'src_map', src_map.to(device))
                setattr(self, 'alignment', align.to(device))

                ex_vocabs = [x[-1] for x in data]
                setattr(self, 'src_vocabs', ex_vocabs)

    def __len__(self):
        return self.batch_size

    def generate_summ_bow(self, data, idf_info):

        vocab_size = idf_info["voc_size"]
        all_bow = torch.zeros([self.batch_size, vocab_size], dtype=torch.float)
        customer_bow = torch.zeros([self.batch_size, vocab_size], dtype=torch.float)
        agent_bow = torch.zeros([self.batch_size, vocab_size], dtype=torch.float)

        all_file_counter = idf_info["all"]
        customer_file_counter = idf_info['customer']
        agent_file_counter = idf_info['agent']

        for idx in range(self.batch_size):
            all_counter = data[idx][9]["all"]
            customer_counter = data[idx][9]["customer"]
            agent_counter = data[idx][9]["agent"]

            for key in all_counter.keys():
                all_file_count = all_file_counter[key]
                if not self.args.use_idf:
                    if all_file_count > self.args.max_word_count or \
                      all_file_count < self.args.min_word_count:
                        continue
                all_bow[idx][key] = 1

            for key in customer_counter.keys():
                customer_file_count = customer_file_counter[key]
                if not self.args.use_idf:
                    if customer_file_count > self.args.max_word_count or \
                      customer_file_count < self.args.min_word_count:
                        continue
                customer_bow[idx][key] = 1

            for key in agent_counter.keys():
                agent_file_count = agent_file_counter[key]
                if not self.args.use_idf:
                    if agent_file_count > self.args.max_word_count or \
                      agent_file_count < self.args.min_word_count:
                        continue
                agent_bow[idx][key] = 1

        return all_bow, customer_bow, agent_bow

    def generate_bow(self, data, idf_info):
        vocab_size = idf_info["voc_size"]
        all_bow = torch.zeros([self.batch_size, vocab_size], dtype=torch.float)
        customer_bow = torch.zeros([self.batch_size, vocab_size], dtype=torch.float)
        agent_bow = torch.zeros([self.batch_size, vocab_size], dtype=torch.float)

        all_file_counter = idf_info["all"]
        customer_file_counter = idf_info['customer']
        agent_file_counter = idf_info['agent']
        file_num = idf_info["num"]

        for idx in range(self.batch_size):
            all_counter = data[idx][8]["all"]
            customer_counter = data[idx][8]["customer"]
            agent_counter = data[idx][8]["agent"]

            all_counter_sum = sum(all_counter.values())
            for key, value in all_counter.items():
                all_tf = value / all_counter_sum
                all_file_count = all_file_counter[key]
                if self.args.use_idf:
                    all_idf = math.log(file_num / (all_file_count + 1.))
                else:
                    all_idf = 0. if all_file_count > self.args.max_word_count or \
                        all_file_count < self.args.min_word_count else 1.
                all_bow[idx][key] = all_tf * all_idf

            customer_counter_sum = sum(customer_counter.values())
            for key, value in customer_counter.items():
                customer_tf = value / customer_counter_sum
                customer_file_count = customer_file_counter[key]
                if self.args.use_idf:
                    customer_idf = math.log(file_num / (customer_file_count + 1.))
                else:
                    customer_idf = 0. if customer_file_count > self.args.max_word_count or \
                        customer_file_count < self.args.min_word_count else 1.
                customer_bow[idx][key] = customer_tf * customer_idf

            agent_counter_sum = sum(agent_counter.values())
            for key, value in agent_counter.items():
                agent_tf = value / agent_counter_sum
                agent_file_count = agent_file_counter[key]
                if self.args.use_idf:
                    agent_idf = math.log(file_num / (agent_file_count + 1.))
                else:
                    agent_idf = 0. if agent_file_count > self.args.max_word_count or \
                        agent_file_count < self.args.min_word_count else 1.
                agent_bow[idx][key] = agent_tf * agent_idf

        return all_bow, customer_bow, agent_bow

    def make_src_map(self, data):
        src_size = max([len(t) for t in data])
        src_vocab_size = max([max(t) for t in data]) + 1
        src_map = torch.zeros(len(data), src_size, src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                src_map[i, j, t] = 1
        return src_map


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "val", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def batch_size_fn(new, count):
    tgt = new[5]
    global max_n_tokens
    if count == 1:
        max_n_tokens = 0
    max_n_tokens = max(max_n_tokens, len(tgt))
    src_elements = count * max_n_tokens
    if (count > 6):
        return src_elements + 1e3
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size, batch_ex_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.batch_ex_size = batch_ex_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        if self.args.topic_model:
            self.idf_info = torch.load(args.idf_info_path)
        else:
            self.idf_info = None
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args, dataset=self.cur_dataset, batch_size=self.batch_size,
                            batch_ex_size=self.batch_ex_size, device=self.device, shuffle=self.shuffle,
                            is_test=self.is_test, idf_info=self.idf_info)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, batch_ex_size,
                 device=None, is_test=False, shuffle=True, idf_info=None):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.max_ex_num = batch_ex_size
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle
        self.sort_key = lambda x: len(x[1])
        self.idf_info = idf_info

        self._iterations_this_epoch = 0
        self.batch_size_fn = batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):

        src_session = []
        segs_session = []
        txt_session = []

        session = ex["session"]
        dialogue = ex["dialogue"]
        topic_info = ex["topic_info"]

        if "summary" in ex.keys():
            tgt = ex["summary"]["id"][:self.args.max_tgt_len][:-1]+[2]
            tgt_txt = ex["summary"]["content_tokens"]
            if "ex_labels" in ex["summary"].keys():
                tgt_labels = ex["summary"]["ex_labels"]
                topic_summ_info = ex["summary"]["topic_summ_info"]
                if len(tgt_labels) == 0:
                    return None
            else:
                tgt_labels, topic_summ_info = None, None
        else:
            tgt, tgt_txt, tgt_labels, topic_summ_info = None, None, None, None

        src_ex = dialogue['src_id']
        segs_ex = dialogue['segs']

        end_id = [src_ex[-1]]
        src_ex = src_ex[:-1][:self.args.max_pos - 1] + end_id
        segs_ex = segs_ex[:self.args.max_pos]

        if self.args.copy_attn:

            # build dynamic dict
            ex_vocab = Vocab(Counter(src_ex), specials=[0])

            src_map = [ex_vocab.stoi[w] for w in src_ex]

            if tgt is not None:
                align = [0] + [ex_vocab.stoi[w] if w in ex_vocab.stoi.keys() else 0 for w in tgt[1:-1]] + [0]
            else:
                align = None

        for turn in session:
            index = turn['index']
            src = turn['src_id']
            segs = turn['segs']
            original_txt = turn['original_txt']
            role = turn['role']
            end_id = [src[-1]]
            src = src[:-1][:self.args.max_pos - 1] + end_id
            segs = segs[:self.args.max_pos]
            if role == '客服':
                original_txt = "(" + str(index) + ') 【客服】 ' + original_txt
            else:
                original_txt = "(" + str(index) + ') 【客户】 ' + original_txt

            src_session.append(src)
            segs_session.append(segs)
            txt_session.append(original_txt)

        if self.args.copy_attn:
            return src_ex, segs_ex, src_session, segs_session, txt_session, \
                tgt, tgt_txt, tgt_labels, topic_info, topic_summ_info, src_map, align, ex_vocab
        else:
            return src_ex, segs_ex, src_session, segs_session, txt_session, \
                tgt, tgt_txt, tgt_labels, topic_info, topic_summ_info

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size, max_ex_num=5):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
            if len(minibatch) >= max_ex_num:
                yield minibatch
                minibatch, size_so_far = [], 0
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            p_batch = self.batch(buffer, self.batch_size, self.max_ex_num)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(self.args, minibatch, self.device, self.is_test, self.idf_info)

                yield batch
            return
