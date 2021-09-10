import copy
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

from models.topic import MultiTopicModel
from models.decoder_tf import TransformerDecoder
from models.encoder import Bert, TransformerEncoder
from models.generator import Generator
from others.utils import tile
from others.vocab_wrapper import VocabWrapper


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
            self.encoder = TransformerEncoder(self.hidden_size, args.enc_ff_size, args.enc_heads,
                                              args.enc_dropout, args.enc_layers)
            tgt_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)

        self.hier_encoder = TransformerEncoder(self.hidden_size, args.hier_ff_size, args.hier_heads,
                                               args.hier_dropout, args.hier_layers)

        topic_emb_size = self.args.word_emb_size * 3
        # topic_emb_size = self.args.word_emb_size*6 if self.args.split_noise else self.args.word_emb_size*3
        self.decoder = TransformerDecoder(
            self.args.dec_layers, self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout,
            embeddings=tgt_embeddings, topic=self.args.topic_model,
            topic_dim=topic_emb_size, split_noise=self.args.split_noise)

        self.generator = Generator(self.vocab_size, self.args.dec_hidden_size, self.pad_token)

        self.generator.linear.weight = self.decoder.embeddings.weight

        # Topic Model
        if self.args.topic_model:
            # Using Golve or Word2vec embedding
            if self.args.tokenize:
                self.voc_wrapper = VocabWrapper(self.args.word_emb_mode)
                self.voc_wrapper.load_emb(self.args.word_emb_path)
                self.voc_emb = torch.tensor(self.voc_wrapper.get_emb())
            else:
                self.voc_emb = torch.empty(self.vocab_size, self.args.word_emb_size)
                xavier_uniform_(self.voc_emb)
                # self.voc_emb.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
            self.topic_model = MultiTopicModel(self.voc_emb.size(0), self.voc_emb.size(-1),
                                               args.topic_num, args.noise_rate, self.voc_emb,
                                               agent=args.agent, cust=args.cust)
            # self.topic_linear = nn.Linear(topic_emb_size, self.hidden_size)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
            if args.share_emb:
                self.generator.linear.weight = self.decoder.embeddings.weight
        else:
            # initialize params.
            if args.encoder == "transformer":
                for module in self.encoder.modules():
                    self._set_parameter_tf(module)
            for module in self.decoder.modules():
                self._set_parameter_tf(module)
            for module in self.hier_encoder.modules():
                self._set_parameter_tf(module)
            for p in self.generator.parameters():
                self._set_parameter_linear(p)
            # if self.args.topic_model:
            #     for p in self.topic_linear.parameters():
            #         self._set_parameter_linear(p)
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

    def _fast_translate_batch(self, batch, memory_bank, max_length, init_tokens=None, memory_mask=None,
                              min_length=2, beam_size=3, topic_vec=None):

        batch_size = memory_bank.size(0)

        dec_states = self.decoder.init_decoder_state(batch.src, memory_bank, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=0)
        init_tokens = tile(init_tokens, beam_size, dim=0)
        memory_mask = tile(memory_mask, beam_size, dim=0)
        if self.args.split_noise:
            topic_vec = (tile(topic_vec[0], beam_size, dim=0),
                         tile(topic_vec[1], beam_size, dim=0))
        else:
            topic_vec = tile(topic_vec, beam_size, dim=0)

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
            if step > 0:
                init_tokens = None
            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states, _ = self.decoder(decoder_input, memory_bank, dec_states, init_tokens, step=step,
                                                  memory_masks=memory_mask, topic_vec=topic_vec)

            # Generator forward.
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
            if init_tokens is not None:
                init_tokens = init_tokens.index_select(0, select_indices)
            if topic_vec is not None:
                if self.args.split_noise:
                    topic_vec = (topic_vec[0].index_select(0, select_indices),
                                 topic_vec[1].index_select(0, select_indices))
                else:
                    topic_vec = topic_vec.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        results = [t[0] for t in results]
        return results

    def _topic_vec_gen(self, batch, topic_info):

        src, ex_segs = batch.src, batch.ex_segs
        bsz, max_len = len(ex_segs), max(ex_segs)
        topic_vec_all, topic_vec_cust, topic_vec_agent = topic_info
        customer = (topic_vec_cust is not None)
        agent = (topic_vec_agent is not None)

        if customer:
            cust_mask = torch.split(src[:, 1].eq(self.customer_token), ex_segs)
            cust_mask = pad_sequence(cust_mask, batch_first=True, padding_value=0).float()
        if agent:
            agent_mask = torch.split(src[:, 1].eq(self.agent_token), ex_segs)
            agent_mask = pad_sequence(agent_mask, batch_first=True, padding_value=0).float()

        if agent and customer:
            if self.args.split_noise:
                topic_vec_agent_summ, topic_vec_agent_noise = topic_vec_agent
                topic_vec_cust_summ, topic_vec_cust_noise = topic_vec_cust
                topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
                topic_vec_summ = torch.cat([topic_vec_agent_summ.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
                                            topic_vec_cust_summ.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
                                            topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)], -1)
                topic_vec_noise = torch.cat([topic_vec_agent_noise.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
                                             topic_vec_cust_noise.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
                                             topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)], -1)
                topic_vec = (topic_vec_summ, topic_vec_noise)
            else:
                topic_vec = torch.cat([topic_vec_agent.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
                                       topic_vec_cust.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
                                       topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)], -1)
        elif agent:
            if self.args.split_noise:
                topic_vec_agent_summ, topic_vec_agent_noise = topic_vec_agent
                topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
                topic_vec_summ = torch.cat([topic_vec_agent_summ.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
                                            topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)], -1)
                topic_vec_noise = torch.cat([topic_vec_agent_noise.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
                                             topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)], -1)
                topic_vec = (topic_vec_summ, topic_vec_noise)
            else:
                topic_vec = torch.cat([topic_vec_agent.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
                                       topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)], -1)
        elif customer:
            if self.args.split_noise:
                topic_vec_cust_summ, topic_vec_customer_noise = topic_vec_cust
                topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
                topic_vec_summ = torch.cat([topic_vec_cust_summ.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
                                            topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)], -1)
                topic_vec_noise = torch.cat([topic_vec_cust_noise.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
                                            topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)], -1)
                topic_vec = (topic_vec_summ, topic_vec_noise)
            else:
                topic_vec = torch.cat([topic_vec_cust.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
                                       topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)], -1)
        else:
            if self.args.split_noise:
                topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
                topic_vec_summ = topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)
                topic_vec_noise = topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)
                topic_vec = (topic_vec_summ, topic_vec_noise)
            else:
                topic_vec = topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)

        return topic_vec

    def forward(self, batch):

        src = batch.src
        tgt = batch.tgt
        segs = batch.segs
        mask_src = batch.mask_src
        ex_segs = batch.ex_segs

        if self.args.encoder == "bert":
            top_vec = self.encoder(src, segs, mask_src)
        else:
            src_emb = self.embeddings(src)
            top_vec = self.encoder(src_emb, 1-mask_src)
        clss = top_vec[:, 0, :]

        # Hierarchical encoder
        cls_list = torch.split(clss, ex_segs)
        cls_input = pad_sequence(cls_list, batch_first=True, padding_value=0.)
        cls_mask_list = [mask_src.new_zeros([length]) for length in ex_segs]
        cls_mask = pad_sequence(cls_mask_list, batch_first=True, padding_value=1)

        hier = self.hier_encoder(cls_input, cls_mask)
        # hier = hier.view(-1, hier.size(-1))[(1-cls_mask.view(-1)).byte()]

        if self.args.topic_model:
            all_bow, customer_bow, agent_bow = \
                batch.all_bow, batch.customer_bow, batch.agent_bow
            if self.args.split_noise:
                summ_all, summ_customer, summ_agent = \
                    batch.summ_all, batch.summ_customer, batch.summ_agent
            else:
                summ_all, summ_customer, summ_agent = None, None, None
            topic_loss, topic_info = self.topic_model(all_bow, customer_bow, agent_bow,
                                                      summ_all, summ_customer, summ_agent)
            topic_loss = self.args.loss_lambda * topic_loss
            # topic_dec_init = self.topic_linear(torch.cat(topic_info, -1))
            topic_vec = self._topic_vec_gen(batch, topic_info)
        else:
            # topic_loss, topic_dec_init, topic_vec = None, None, None
            topic_loss, topic_vec = None, None

        if self.training:
            dec_state = self.decoder.init_decoder_state(src, hier)
            decode_output, _, _ = self.decoder(tgt[:, :-1], hier, dec_state, memory_masks=cls_mask,
                                               topic_vec=topic_vec)
            summary = None
        else:
            decode_output = None
            summary = self._fast_translate_batch(batch, hier, self.max_length, memory_mask=cls_mask,
                                                 min_length=2, beam_size=self.beam_size,
                                                 topic_vec=topic_vec)

        return decode_output, summary, topic_loss
