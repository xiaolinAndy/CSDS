import torch
import torch.nn as nn
from models.neural import aeq
from models.neural import gumbel_softmax


class Generator(nn.Module):
    def __init__(self, vocab_size, dec_hidden_size, pad_idx):
        super(Generator, self).__init__()
        self.linear = nn.Linear(dec_hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.pad_idx = pad_idx

    def forward(self, x, use_gumbel_softmax=False):
        output = self.linear(x)
        output[:, self.pad_idx] = -float('inf')
        if use_gumbel_softmax:
            output = gumbel_softmax(output, log_mode=True, dim=-1)
        else:
            output = self.softmax(output)
        return output


class PointerNetGenerator(nn.Module):
    def __init__(self, mem_hidden_size, dec_hidden_size, hidden_size):
        super(PointerNetGenerator, self).__init__()
        self.terminate_state = nn.Parameter(torch.empty(1, mem_hidden_size))
        self.linear_dec = nn.Linear(dec_hidden_size, hidden_size)
        self.linear_mem = nn.Linear(mem_hidden_size, hidden_size)
        self.score_linear = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, mem, dec_hid, mem_mask, dec_mask, dup_mask):

        batch_size = mem.size(0)

        # Add terminate state
        mem = torch.cat([self.terminate_state.unsqueeze(0).expand(batch_size, 1, -1), mem], 1)
        mem_mask = torch.cat([torch.zeros([batch_size, 1], dtype=mem_mask.dtype, device=mem_mask.device), mem_mask], 1)

        mem_len = mem.size(1)
        dec_len = dec_hid.size(1)

        # batch * dec_len * mem_len * hid_size
        mem_expand = mem.unsqueeze(1).expand(batch_size, dec_len, mem_len, -1)
        dec_expand = dec_hid.unsqueeze(2).expand(batch_size, dec_len, mem_len, -1)
        mask_expand = mem_mask.unsqueeze(1).expand(batch_size, dec_len, mem_len)
        score = self.score_linear(self.tanh(self.linear_mem(mem_expand) + self.linear_dec(dec_expand))).squeeze_(-1)
        score[mask_expand] = -float('inf')

        # Avoid duplicate extraction.
        dup_mask[dec_mask, :] = 0
        if score.requires_grad:
            dup_mask = dup_mask.float()
            dup_mask[dup_mask == 1] = -float('inf')
            score = dup_mask + score
        else:
            score[dup_mask.byte()] = -float('inf')

        output = self.softmax(score)
        return output


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.
    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       output_size (int): size of output representation
    """

    def __init__(self, output_size, input_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.padding_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        batch, slen_, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.padding_idx] = -float('inf')
        prob = self.softmax(logits)

        # Probability of copying p(z=1) batch.
        p_copy = self.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy.expand_as(prob))
        mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(batch, -1, slen), src_map)
        copy_prob = copy_prob.view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


def collapse_copy_scores(scores, batch, tgt_vocab, batch_index=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambigious.
    """
    offset = len(tgt_vocab)
    for b in range(scores.size(0)):
        blank = []
        fill = []

        if batch_index is not None:
            src_vocab = batch.src_vocabs[batch_index[b]]
        else:
            src_vocab = batch.src_vocabs[b]

        for i in range(1, len(src_vocab)):
            ti = src_vocab.itos[i]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.tensor(blank, device=scores.device)
            fill = torch.tensor(fill, device=scores.device)
            scores[b, :].index_add_(1, fill,
                                    scores[b, :].index_select(1, blank))
            scores[b, :].index_fill_(1, blank, 1e-10)
    return scores
