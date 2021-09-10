# encoding=utf-8

import argparse
from others.logging import init_logger
from prepro import data_builder as data_builder


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-type", default='train', type=str)
    parser.add_argument("-raw_path", default='json_data')
    parser.add_argument("-save_path", default='bert_data')
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-idlize", nargs='?', const=True, default=False)

    parser.add_argument("-bert_dir", default='bert/chinese_rbt3')
    parser.add_argument('-min_src_ntokens', default=1, type=int)
    parser.add_argument('-max_src_ntokens', default=3000, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=50, type=int)
    parser.add_argument('-min_tgt_ntokens', default=1, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    parser.add_argument('-min_turns', default=1, type=int)
    parser.add_argument('-max_turns', default=100, type=int)
    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-tokenize", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-emb_mode", default="word2vec", type=str, choices=["glove", "word2vec"])
    parser.add_argument("-emb_path", default="", type=str)
    parser.add_argument("-ex_max_token_num", default=500, type=int)
    parser.add_argument("-truncated", nargs='?', const=True, default=False)
    parser.add_argument("-add_ex_label", nargs='?', const=True, default=False)

    parser.add_argument('-log_file', default='logs/preprocess.log')
    parser.add_argument('-dataset', default='')

    args = parser.parse_args()
    if args.type not in ["train", "dev", "test"]:
        print("Invalid data type! Data type should be 'train', 'dev', or 'test'.")
        exit(0)
    init_logger(args.log_file)
    data_builder.format_to_bert(args)
