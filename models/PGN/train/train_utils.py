from torch.autograd import Variable
import numpy as np
import torch
from utils import config


def padded_sequence(seqs, pad):
    max_len = max([len(seq) for seq in seqs])
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    length = [len(seq) for seq in seqs]
    return padded_seqs, length

def get_input_from_batch(features, device, vocab):
    batch_size = len(features[0][0])
    enc_input_ids, enc_lens, enc_batch_extend_vocab, art_oovs, max_art_oovs = features[0]
    enc_padding_mask = enc_input_ids.ne(vocab.token2idx('<PAD>')).float()

    extra_zeros = None

    if config.pointer_gen:
        # max_art_oovs is the max over all the article oov list in the batch
        if max_art_oovs > 0:
          extra_zeros = torch.zeros((batch_size, max_art_oovs))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_input_ids.size())

    enc_input_ids = enc_input_ids.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    enc_lens = enc_lens.to(device)


    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(device)
    if extra_zeros is not None:
        extra_zeros = extra_zeros.to(device)
    c_t_1 = c_t_1.to(device)

    if coverage is not None:
        coverage = coverage.to(device)

    return enc_input_ids, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(features, device, vocab):
    dec_input_ids, dec_output_ids, dec_lens, sum_sents = features[1]
    dec_padding_mask = dec_input_ids.ne(vocab.token2idx('<PAD>')).float()
    max_dec_len = max(dec_lens)

    dec_input_ids = dec_input_ids.to(device)
    dec_padding_mask = dec_padding_mask.to(device)
    dec_lens = dec_lens.to(device)
    dec_output_ids = dec_output_ids.to(device)

    return dec_input_ids, dec_padding_mask, max_dec_len, dec_lens, dec_output_ids, sum_sents

