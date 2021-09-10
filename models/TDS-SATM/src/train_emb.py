import argparse
import os
import json
import glob
from os.path import join as pjoin
from others.vocab_wrapper import VocabWrapper


def train_emb(args):
    data_dir = os.path.abspath(args.data_path)
    print("Preparing to process %s ..." % data_dir)
    raw_files = glob.glob(pjoin(data_dir, '*.json'))

    ex_num = 0
    vocab_wrapper = VocabWrapper(args.mode, args.emb_size)
    vocab_wrapper.init_model()

    file_ex = []
    for s in raw_files:
        exs = json.load(open(s))
        print("Processing File " + s)
        for ex in exs:
            example = list(map(lambda x: x['word'], ex['session']))
            file_ex.extend(example)
            ex_num += 1
    vocab_wrapper.train(file_ex)
    vocab_wrapper.report()
    print("Datasets size: %d" % ex_num)
    vocab_wrapper.save_emb(args.emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='word2vec', type=str, choices=['glove', 'word2vec'])
    parser.add_argument("-data_path", default="", type=str)
    parser.add_argument("-emb_size", default=100, type=int)
    parser.add_argument("-emb_path", default="", type=str)

    args = parser.parse_args()

    train_emb(args)
