import copy
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_

from models.decoder_tf import TransformerDecoder
from models.decoder_rnn import RNNDecoder
from models.encoder import Bert, TransformerEncoder, RNNEncoder
from models.generator import Generator, CopyGenerator, collapse_copy_scores
from others.utils import tile


class Model(nn.Module):
    def __init__(self, args, device, vocab, checkpoint=None):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.beam_size = args.beam_size
        self.max_length = args.max_length
        self.min_length = args.min_length

        # special tokens
        self.start_token = vocab['[unused1]']
        self.end_token = vocab['[unused2]']
        self.pad_token = vocab['[PAD]']
        self.mask_token = vocab['[MASK]']
        self.seg_token = vocab['[SEP]']
        self.cls_token = vocab['[CLS]']
        self.agent_token = vocab['[unused3]']
        self.customer_token = vocab['[unused4]']

        if args.encoder == 'bert':
            self.encoder = Bert(args.bert_dir, args.finetune_bert)
            if(args.max_pos > 512):
                my_pos_embeddings = nn.Embedding(args.max_pos, self.encoder.model.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = self.encoder.model.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = self.encoder.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(args.max_pos-512, 1)
                self.encoder.model.embeddings.position_embeddings = my_pos_embeddings
            self.hidden_size = self.encoder.model.config.hidden_size
            tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
        else:
            self.hidden_size = args.enc_hidden_size
            self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
            tgt_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
            if args.encoder == 'rnn':
                self.encoder = RNNEncoder('LSTM', bidirectional=True, num_layers=args.enc_layers,
                                          hidden_size=self.hidden_size, dropout=args.enc_dropout,
                                          embeddings=self.embeddings)
            elif args.encoder == "transformer":
                self.encoder = TransformerEncoder(self.hidden_size, args.enc_ff_size, args.enc_heads,
                                                  args.enc_dropout, args.enc_layers)

        if args.decoder == "transformer":
            self.decoder = TransformerDecoder(
                args.dec_layers, args.dec_hidden_size, heads=args.dec_heads,
                d_ff=args.dec_ff_size, dropout=args.dec_dropout,
                embeddings=tgt_embeddings)
        elif args.decoder == "rnn":
            self.decoder = RNNDecoder("LSTM", True, args.dec_layers, args.dec_hidden_size,
                                      dropout=args.dec_dropout, embeddings=tgt_embeddings,
                                      coverage_attn=args.coverage, copy_attn=args.copy_attn)

        if args.copy_attn:
            self.generator = CopyGenerator(self.vocab_size, args.dec_hidden_size, self.pad_token)
        else:
            self.generator = Generator(self.vocab_size, args.dec_hidden_size, self.pad_token)

        self.generator.linear.weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
            if args.share_emb:
                if args.encoder == 'bert':
                    self.embeddings = self.encoder.model.embeddings.word_embeddings
                self.generator.linear.weight = self.decoder.embeddings.weight
        else:
            # initialize params.
            if args.encoder == "transformer":
                for module in self.encoder.modules():
                    self._set_parameter_tf(module)
            elif args.encoder == "rnn":
                for p in self.encoder.parameters():
                    self._set_parameter_linear(p)
            for module in self.decoder.modules():
                self._set_parameter_tf(module)
            for p in self.generator.parameters():
                self._set_parameter_linear(p)
            if args.share_emb:
                if args.encoder == 'bert':
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
                    self.decoder.embeddings = tgt_embeddings
                self.generator.linear.weight = self.decoder.embeddings.weight

        self.to(device)

    def _set_parameter_tf(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()

    def _fast_translate_batch(self, batch, memory_bank, max_length, memory_mask=None,
                              min_length=2, beam_size=3, hidden_state=None, copy_attn=False):

        batch_size = memory_bank.size(0)

        if self.args.decoder == "rnn":
            dec_states = self.decoder.init_decoder_state(batch.src, memory_bank, hidden_state)
        else:
            dec_states = self.decoder.init_decoder_state(batch.src, memory_bank, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=0)
        memory_mask = tile(memory_mask, beam_size, dim=0)
        if copy_attn:
            src_map = tile(batch.src_map, beam_size, dim=0)
        else:
            src_map = None

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=self.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=self.device)

        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=self.device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=self.device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = [[] for _ in range(batch_size)]  # noqa: F812

        for step in range(max_length):
            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            if self.args.decoder == "rnn":
                dec_out, dec_states, attn = self.decoder(decoder_input, memory_bank, dec_states, step=step,
                                                         memory_masks=memory_mask)
            else:
                dec_out, dec_states, attn = self.decoder(decoder_input, memory_bank, dec_states, step=step,
                                                         memory_masks=memory_mask, requires_att=copy_attn)

            # Generator forward.
            if copy_attn:
                probs = self.generator(dec_out.transpose(0, 1).squeeze(0),
                                       attn['copy'].transpose(0, 1).squeeze(0),
                                       src_map)
                probs = collapse_copy_scores(probs.unsqueeze(1), batch, self.vocab,
                                             tile(batch_offset, beam_size, dim=0))
                log_probs = probs.squeeze(1)[:, :self.vocab_size].log()
            else:
                log_probs = self.generator(dec_out.transpose(0, 1).squeeze(0))

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if(cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        if(len(words) <= 3):
                            continue
                        trigrams = [(words[i-1], words[i], words[i+1]) for i in range(1, len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            log_probs[i] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.args.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        _, pred = best_hyp[0]
                        results[b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            if memory_bank is not None:
                memory_bank = memory_bank.index_select(0, select_indices)
            if memory_mask is not None:
                memory_mask = memory_mask.index_select(0, select_indices)
            if src_map is not None:
                src_map = src_map.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        results = [t[0] for t in results]
        return results

    def forward(self, batch):

        src = batch.src
        tgt = batch.tgt
        segs = batch.segs
        mask_src = batch.mask_src

        if self.args.encoder == "bert":
            top_vec = self.encoder(src, segs, mask_src)
        elif self.args.encoder == "transformer":
            src_emb = self.embeddings(src)
            top_vec = self.encoder(src_emb, 1-mask_src)
        elif self.args.encoder == "rnn":
            top_vec, hid = self.encoder(src, mask_src)

        if self.training:
            if self.args.decoder == "rnn":
                dec_state = self.decoder.init_decoder_state(src, top_vec, hid)
                decode_output, _, attn = self.decoder(tgt[:, :-1], top_vec, dec_state, memory_masks=mask_src)
            else:
                dec_state = self.decoder.init_decoder_state(src, top_vec)
                decode_output, _, attn = self.decoder(tgt[:, :-1], top_vec, dec_state, memory_masks=1-mask_src,
                                                      requires_att=self.args.copy_attn)
            summary = None
        else:
            decode_output, attn = None, None
            if self.args.decoder == "rnn":
                summary = self._fast_translate_batch(batch, top_vec, self.max_length, memory_mask=mask_src,
                                                     beam_size=self.beam_size, hidden_state=hid,
                                                     copy_attn=self.args.copy_attn)
            else:
                summary = self._fast_translate_batch(batch, top_vec, self.max_length, memory_mask=1-mask_src,
                                                     beam_size=self.beam_size, copy_attn=self.args.copy_attn)

        return decode_output, summary, attn
