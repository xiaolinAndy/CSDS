import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pytorch_transformers import BertModel
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, rnn_factory


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class Bert(nn.Module):
    def __init__(self, temp_dir, finetune=False):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):

        pe = torch.zeros(max_len, dim)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        position = torch.arange(0, max_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None, add_emb=None):
        emb = emb * math.sqrt(self.dim)
        if add_emb is not None:
            emb = emb + add_emb
        if (step):
            pos = self.pe[:, step][:, None, :]
            emb = emb + pos
        else:
            pos = self.pe[:, :emb.size(1)]
            emb = emb + pos
        emb = self.dropout(emb)
        return emb


class DistancePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        mid_pos = max_len // 2
        # absolute position embedding
        ape = torch.zeros(max_len, dim // 2)
        # distance position embedding
        dpe = torch.zeros(max_len, dim // 2)

        ap = torch.arange(0, max_len).unsqueeze(1)
        dp = torch.abs(torch.arange(0, max_len).unsqueeze(1) - mid_pos)

        div_term = torch.exp((torch.arange(0, dim//2, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim * 2)))
        ape[:, 0::2] = torch.sin(ap.float() * div_term)
        ape[:, 1::2] = torch.cos(ap.float() * div_term)
        dpe[:, 0::2] = torch.sin(dp.float() * div_term)
        dpe[:, 1::2] = torch.cos(dp.float() * div_term)

        ape = ape.unsqueeze(0)
        super(DistancePositionalEncoding, self).__init__()
        self.register_buffer('ape', ape)
        self.register_buffer('dpe', dpe)
        self.dim = dim
        self.mid_pos = mid_pos

    def forward(self, emb, shift):
        device = emb.device
        _, length, _ = emb.size()
        pe_seg = [len(ex) for ex in shift]
        medium_pos = [torch.cat([torch.tensor([0], device=device),
                                 (ex[1:] + ex[:-1]) // 2 + 1,
                                 torch.tensor([length], device=device)], 0)
                      for ex in shift]
        shift = torch.cat(shift, 0)
        index = torch.arange(self.mid_pos, self.mid_pos + length, device=device).\
            unsqueeze(0).expand(len(shift), length) - shift.unsqueeze(1)
        index = torch.split(index, pe_seg)
        dp_index = []
        for i in range(len(index)):
            dpi = torch.zeros([length], device=device)
            for j in range(len(index[i])):
                dpi[medium_pos[i][j]:medium_pos[i][j+1]] = index[i][j][medium_pos[i][j]:medium_pos[i][j+1]]
            dp_index.append(dpi.unsqueeze(0))
        dp_index = torch.cat(dp_index, 0).long()

        dpe = self.dpe[dp_index]
        ape = self.ape[:, :emb.size(1)].expand(emb.size(0), emb.size(1), -1)
        pe = torch.cat([dpe, ape], -1)
        emb = emb + pe
        return emb


class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        mid_pos = max_len // 2
        # relative position embedding
        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len).unsqueeze(1) - mid_pos

        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        super(RelativePositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim
        self.mid_pos = mid_pos

    def forward(self, emb, shift):
        device = emb.device
        bsz, length, _ = emb.size()
        index = torch.arange(self.mid_pos, self.mid_pos + emb.size(1), device=device).\
            unsqueeze(0).expand(bsz, length) - shift.unsqueeze(1)
        pe = self.pe[index]
        emb = emb + pe
        return emb

    def get_emb(self, emb, shift):
        device = emb.device
        index = torch.arange(self.mid_pos, self.mid_pos + emb.size(1), device=device).\
            unsqueeze(0).expand(emb.size(0), emb.size(1)) - shift.unsqueeze(1)
        return self.pe[index]


class RNNEncoder(nn.Module):
    """ A generic recurrent neural network encoder.
    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn = rnn_factory(rnn_type,
                               input_size=embeddings.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               batch_first=True)

    def forward(self, src, mask):

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()
        lengths = mask.sum(dim=1)

        # Lengths data is wrapped inside a Tensor.
        lengths_list = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths_list, batch_first=True, enforce_sorted=False)

        memory_bank, encoder_final = self.rnn(packed_emb)

        memory_bank = unpack(memory_bank, batch_first=True)[0]

        return memory_bank, encoder_final


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type='self')
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerEncoder, self).__init__()
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        x = self.pos_emb(top_vecs)

        for i in range(self.num_inter_layers):
            x = self.transformer[i](i, x, mask)  # all_sents * max_tokens * dim

        output = self.layer_norm(x)

        return output
