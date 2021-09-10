import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from rouge import Rouge
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.categorical import Categorical

from models.topic import MultiTopicModel
from models.decoder_tf import TransformerDecoder
from models.encoder import Bert, TransformerEncoder
from models.generator import Generator, PointerNetGenerator
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

        self.hidden_size = args.enc_hidden_size
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)

        if args.encoder == 'bert':
            self.encoder = Bert(args.bert_dir, args.finetune_bert)
            if(args.max_pos > 512):
                my_pos_embeddings = nn.Embedding(args.max_pos, self.encoder.model.config.hidden_size)
                my_pos_embeddings.weight.data[:512] = self.encoder.model.embeddings.position_embeddings.weight.data
                my_pos_embeddings.weight.data[512:] = self.encoder.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(args.max_pos-512, 1)
                self.encoder.model.embeddings.position_embeddings = my_pos_embeddings
            tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
        else:
            self.encoder = TransformerEncoder(self.hidden_size, args.enc_ff_size, args.enc_heads,
                                              args.enc_dropout, args.enc_layers)
            tgt_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)

        self.sent_encoder = TransformerEncoder(self.hidden_size, args.enc_ff_size, args.enc_heads,
                                               args.enc_dropout, args.enc_layers)
        self.hier_encoder = TransformerEncoder(self.hidden_size, args.hier_ff_size, args.hier_heads,
                                               args.hier_dropout, args.hier_layers)

        if args.cust and args.agent:
            topic_emb_size = self.args.word_emb_size * 3
        elif args.cust or args.agent:
            topic_emb_size = self.args.word_emb_size * 2
        else:
            topic_emb_size = self.args.word_emb_size * 1
        # topic_emb_size = self.args.word_emb_size*6 if self.args.split_noise else self.args.word_emb_size*3
        self.pn_decoder = TransformerDecoder(
            self.args.pn_layers, self.args.pn_hidden_size, heads=self.args.pn_heads,
            d_ff=self.args.pn_ff_size, dropout=self.args.pn_dropout, topic=self.args.topic_model,
            topic_dim=topic_emb_size, split_noise=self.args.split_noise
        )

        self.pn_generator = PointerNetGenerator(self.hidden_size, self.hidden_size, self.hidden_size)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout,
            embeddings=tgt_embeddings, topic=self.args.topic_model,
            topic_dim=topic_emb_size, split_noise=self.args.split_noise
        )

        self.generator = Generator(self.vocab_size, self.args.dec_hidden_size, self.pad_token)

        self.generator.linear.weight = self.decoder.embeddings.weight

        # Topic Model
        if self.args.topic_model:
            # Using Golve or Word2vec embedding
            if self.args.tokenize:
                self.voc_wrapper = VocabWrapper(self.args.word_emb_mode)
                self.voc_wrapper.load_emb(self.args.word_emb_path)
                voc_emb = torch.tensor(self.voc_wrapper.get_emb())
            else:
                voc_emb = torch.empty(self.vocab_size, self.args.word_emb_size)
            self.topic_model = MultiTopicModel(voc_emb.size(0), voc_emb.size(-1), args.topic_num, args.noise_rate,
                                               voc_emb, agent=args.agent, cust=args.cust)
            if self.args.split_noise:
                self.topic_gate_linear_summ = nn.Linear(self.hidden_size + topic_emb_size, topic_emb_size)
                self.topic_emb_linear_summ = nn.Linear(self.hidden_size, topic_emb_size)
                self.topic_gate_linear_noise = nn.Linear(self.hidden_size + topic_emb_size, topic_emb_size)
                self.topic_emb_linear_noise = nn.Linear(self.hidden_size, topic_emb_size)
            else:
                self.topic_gate_linear = nn.Linear(self.hidden_size + topic_emb_size, topic_emb_size)
                self.topic_emb_linear = nn.Linear(self.hidden_size, topic_emb_size)
        # else:
        self.pn_init_token = nn.Parameter(torch.empty([1, self.hidden_size]))

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
            for module in self.decoder.modules():
                self._set_parameter_tf(module)
            for module in self.sent_encoder.modules():
                self._set_parameter_tf(module)
            for module in self.hier_encoder.modules():
                self._set_parameter_tf(module)
            for module in self.pn_decoder.modules():
                self._set_parameter_tf(module)
            for p in self.pn_generator.parameters():
                self._set_parameter_linear(p)
            for p in self.generator.parameters():
                self._set_parameter_linear(p)
            if self.args.topic_model:
                if self.args.split_noise:
                    for p in self.topic_gate_linear_summ.parameters():
                        self._set_parameter_linear(p)
                    for p in self.topic_emb_linear_summ.parameters():
                        self._set_parameter_linear(p)
                    for p in self.topic_gate_linear_noise.parameters():
                        self._set_parameter_linear(p)
                    for p in self.topic_emb_linear_noise.parameters():
                        self._set_parameter_linear(p)
                else:
                    for p in self.topic_gate_linear.parameters():
                        self._set_parameter_linear(p)
                    for p in self.topic_emb_linear.parameters():
                        self._set_parameter_linear(p)
            self._set_parameter_linear(self.pn_init_token)
            if args.share_emb:
                if args.encoder == 'bert':
                    self.embeddings = self.encoder.model.embeddings.word_embeddings
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

    def _index_2_onehot(self, idx_tgt, ex_segs):
        onehot_tgt_list = []
        size = max(ex_segs) + 1
        for i in range(len(ex_segs)):
            # Add terminate probability (position 0)
            idx_tgt_extend = torch.tensor([num+1 for num in idx_tgt[i]] + [0], device=self.device)
            one_hot = F.one_hot(idx_tgt_extend, size)
            onehot_tgt_list.append(one_hot)
        one_hot_tgt = pad_sequence(onehot_tgt_list, batch_first=True, padding_value=0)

        # Avoid duplicate extraction.
        dup_mask = torch.zeros_like(one_hot_tgt)
        for i in range(one_hot_tgt.size(0)):
            for j in range(1, one_hot_tgt.size(1)):
                dup_mask[i, j] += (dup_mask[i, j-1] + one_hot_tgt[i, j-1])
        return one_hot_tgt, dup_mask

    def _map_src(self, src, tgt_labels, ex_segs, sep_token=None):
        src_list = torch.split(src, ex_segs)
        src_mapped_list = []
        seg_mapped_list = []
        idx_mapped_list = []
        agent_mask_mapped_list = []
        customer_mask_mapped_list = []
        for idx in range(len(ex_segs)):
            new_src_list = [torch.tensor([self.cls_token], device=self.device)]
            new_seg_list = [torch.tensor([0], device=self.device)]
            new_idx_list = [torch.tensor([-1], device=self.device)]
            if self.args.topic_model:
                agent_mask_list = [torch.tensor([1], dtype=torch.uint8, device=self.device)]
                customer_mask_list = [torch.tensor([1], dtype=torch.uint8, device=self.device)]
            tgt_label_sorted = sorted(tgt_labels[idx])
            src_cand = src_list[idx][tgt_label_sorted]
            # src_cand = src_list[idx][tgt_labels[idx]]
            segment_id = 0
            for j, sent in enumerate(src_cand):
                filted_sent = sent[sent != self.pad_token][1:]
                if sep_token is not None:
                    filted_sent = torch.cat([filted_sent, torch.tensor([sep_token],
                                             device=self.device)], -1)
                new_src_list.append(filted_sent)
                new_seg_list.append(torch.tensor([segment_id] * filted_sent.size(0), device=self.device))
                new_idx_list.append(torch.tensor([tgt_label_sorted[j]] * filted_sent.size(0), device=self.device))
                if self.args.topic_model:
                    if filted_sent[0].item() == self.agent_token:
                        agent_mask_list.append(torch.tensor([1] * filted_sent.size(0), dtype=torch.uint8, device=self.device))
                        customer_mask_list.append(torch.tensor([0] * filted_sent.size(0), dtype=torch.uint8, device=self.device))
                    else:
                        agent_mask_list.append(torch.tensor([0] * filted_sent.size(0), dtype=torch.uint8, device=self.device))
                        customer_mask_list.append(torch.tensor([1] * filted_sent.size(0), dtype=torch.uint8, device=self.device))
                segment_id = 1 - segment_id
            new_src = torch.cat(new_src_list, 0)[:self.args.max_pos]
            new_seg = torch.cat(new_seg_list, 0)[:self.args.max_pos]
            new_idx = torch.cat(new_idx_list, 0)[:self.args.max_pos]
            src_mapped_list.append(new_src)
            seg_mapped_list.append(new_seg)
            idx_mapped_list.append(new_idx)
            if self.args.topic_model:
                agent_mask = torch.cat(agent_mask_list, 0)[:self.args.max_pos]
                customer_mask = torch.cat(customer_mask_list, 0)[:self.args.max_pos]
                agent_mask_mapped_list.append(agent_mask)
                customer_mask_mapped_list.append(customer_mask)

        src_mapped = pad_sequence(src_mapped_list, batch_first=True, padding_value=self.pad_token)
        seg_mapped = pad_sequence(seg_mapped_list, batch_first=True, padding_value=self.pad_token)
        idx_mapped = pad_sequence(idx_mapped_list, batch_first=True, padding_value=-1)
        mask_mapped = src_mapped.data.ne(self.pad_token)

        if self.args.topic_model:
            agent_mask_mapped = pad_sequence(agent_mask_mapped_list, batch_first=True, padding_value=self.pad_token)
            customer_mask_mapped = pad_sequence(customer_mask_mapped_list, batch_first=True, padding_value=self.pad_token)
        else:
            agent_mask_mapped, customer_mask_mapped = None, None

        return src_mapped, seg_mapped, mask_mapped, agent_mask_mapped, customer_mask_mapped, idx_mapped

    def _base_summary_generate(self, batch, topic_info):
        src = batch.src
        mask_src = batch.mask_src
        ex_segs = batch.ex_segs

        # Get base summaries for reward computation.
        # sent encoding
        src_emb = self.embeddings(src)
        sent_hid = self.sent_encoder(src_emb, ~mask_src)[:, 0, :]

        # hierarchical encoding
        sent_list = torch.split(sent_hid, ex_segs)
        sent_input = pad_sequence(sent_list, batch_first=True, padding_value=0.)
        sent_mask_list = [mask_src.new_zeros([length]) for length in ex_segs]
        sent_mask = pad_sequence(sent_mask_list, batch_first=True, padding_value=1)
        hier = self.hier_encoder(sent_input, sent_mask)

        if self.args.topic_model:
            topic_vec_pn = self._topic_vec_pn(batch, topic_info)
        else:
            topic_vec_pn = None

        # base summay generate
        pn_result = self._pointer_net_decoding(batch, hier, self.pn_init_token,
                                               self.args.max_pos, memory_mask=sent_mask,
                                               method="max", topic_vec=topic_vec_pn)
        ext_labels = [t[0][0].tolist() for t in pn_result]
        src_mapped, segs_mapped, mask_src_mapped, agent_mask, customer_mask, idx_mapped = \
            self._map_src(src, ext_labels, ex_segs, self.seg_token)

        # Encoding with generated labels.
        if self.args.encoder == "bert":
            top_vec = self.encoder(src_mapped, segs_mapped, mask_src_mapped)
        else:
            src_mapped_emb = self.embeddings(src_mapped)
            top_vec = self.encoder(src_mapped_emb, ~mask_src_mapped)

        if self.args.topic_model:
            topic_vec_ge = self._topic_vec_ge(topic_info, agent_mask, customer_mask, idx_mapped, hier)
        else:
            topic_vec_ge = None

        summary_base = self._fast_translate_batch(batch, top_vec, self.max_length,
                                                  memory_mask=~mask_src_mapped,
                                                  min_length=2, beam_size=1,
                                                  topic_vec=topic_vec_ge)
        return summary_base

    def _rl(self, batch, pn_result, summary_sample, summary_base, gamma=0.95):

        # pn_result: labels, probs, state_labels (including stopping state)
        probs = torch.cat(list(map(lambda x: x[0][1], pn_result)), 0)

        rouge = Rouge()
        rewards = []
        base_rewards = []
        # calculate rewards
        for i in range(len(batch)):
            gold_summ = " ".join(map(lambda x: str(x.item()), batch.tgt[i][batch.tgt[i] != self.pad_token][1:-1]))
            pred_summ = " ".join(map(lambda x: str(x.item()), summary_sample[i][:-1]))
            base_summ = " ".join(map(lambda x: str(x.item()), summary_base[i][:-1]))
            rouge_score = rouge.get_scores(pred_summ, gold_summ)
            rouge_base = rouge.get_scores(base_summ, gold_summ)
            reward = rouge_score[0]["rouge-l"]["f"]
            reward_base = rouge_base[0]["rouge-l"]["f"]
            # original rewards
            rewards.append(torch.tensor([reward] * pn_result[i][0][2].size(0), device=self.device))
            base_rewards.append(torch.tensor([reward_base] * pn_result[i][0][2].size(0), device=self.device))
        rewards = torch.cat(rewards, 0)
        base_rewards = torch.cat(base_rewards, 0)
        new_rewards = rewards - base_rewards
        rl_loss = torch.sum(-probs * new_rewards)
        return rl_loss

    def _pointer_net_decoding(self, batch, memory_bank, init_tokens, max_length,
                              memory_mask=None, min_length=2, topic_vec=None, method="sample"):

        batch_size, mem_len, _ = memory_bank.size()
        dist_size = mem_len + 1

        dec_states = self.pn_decoder.init_decoder_state(batch.src, memory_bank, with_cache=True)

        alive_seq = init_tokens.unsqueeze(0).expand(batch_size, 1, -1)

        dup_mask = memory_mask.new_zeros([batch_size, 1, dist_size])
        pred_label = torch.tensor([], device=self.device, dtype=torch.long)
        pred_prob = torch.tensor([], device=self.device)

        # record batch id
        batch_idx = torch.arange(batch_size, device=self.device)

        # extracted sent length
        sent_length = torch.split(batch.src.ne(self.pad_token).sum(1), batch.ex_segs)
        # record extracted sent length
        ext_length = torch.zeros([batch_size], dtype=torch.long, device=self.device)
        # Structure that holds finished hypotheses.
        results = [[] for _ in range(batch_size)]

        for step in range(max_length):

            # Decoder forward.
            decoder_input = alive_seq[:, -1].unsqueeze(1)
            tgt_mask = memory_mask.new_zeros([alive_seq.size(0), 1])

            dec_out, dec_states, _ = self.pn_decoder(decoder_input, memory_bank, dec_states, step=step,
                                                     memory_masks=memory_mask, tgt_masks=tgt_mask, topic_vec=topic_vec)

            # Generator forward.
            log_probs = self.pn_generator(memory_bank, dec_out, memory_mask, tgt_mask, dup_mask)

            if step < min_length:
                if log_probs.requires_grad:
                    mask = torch.zeros_like(log_probs)
                    mask[:, :, 0] = -1e20
                    log_probs = log_probs + mask
                else:
                    log_probs[:, :, 0] = -1e20

            # greedy selection or sampling
            if method == "sample":
                m = Categorical(logits=log_probs)
                ids = m.sample()
                scores = m.log_prob(ids)
            else:
                scores, ids = log_probs.max(dim=-1)

            # Avoid duplicate extraction
            dup_mask += F.one_hot(ids, dist_size).bool()

            # Append last prediction.
            last_pre_hid = torch.cat([memory_bank[i][ids[i]-1].unsqueeze(0) for i in range(ids.size(0))], 0)
            alive_seq = torch.cat([alive_seq, last_pre_hid], 1)

            # Append last pred label and probability
            pred_label = torch.cat([pred_label, ids-1], -1)
            pred_prob = torch.cat([pred_prob, scores], -1)

            # finished if at stop state
            is_finished = ids.eq(0)

            # finished if length exceeds
            for i in range(len(ext_length)):
                ext_length[i] += sent_length[batch_idx[i]][ids[i]-1].item()
                if ext_length[i].item() > max_length:
                    is_finished[i] = 1

            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                for i in range(is_finished.size(0)):
                    # Store finished hypotheses for this batch.
                    # If the batch reached the end, save the results.
                    if end_condition[i]:
                        if pred_label[i, -1].item() == -1:
                            results[batch_idx[i]].append((pred_label[i, :-1], pred_prob[i], pred_label[i]+1))
                        else:
                            results[batch_idx[i]].append((pred_label[i, :-1], pred_prob[i, :-1], pred_label[i, :-1]+1))
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all examples are finished, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                alive_seq = alive_seq.index_select(0, non_finished)
                batch_idx = batch_idx.index_select(0, non_finished)
                pred_label = pred_label.index_select(0, non_finished)
                pred_prob = pred_prob.index_select(0, non_finished)
                memory_bank = memory_bank.index_select(0, non_finished)
                memory_mask = memory_mask.index_select(0, non_finished)
                ext_length = ext_length.index_select(0, non_finished)
                dup_mask = dup_mask.index_select(0, non_finished)
                if topic_vec is not None:
                    if self.args.split_noise:
                        topic_vec = (topic_vec[0].index_select(0, non_finished),
                                     topic_vec[1].index_select(0, non_finished))
                    else:
                        topic_vec = topic_vec.index_select(0, non_finished)
                dec_states.map_batch_fn(
                    lambda state, dim: state.index_select(dim, non_finished))
        return results

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
        hypotheses = [[] for _ in range(batch_size)]

        results = [[] for _ in range(batch_size)]

        for step in range(max_length):
            if step > 0:
                init_tokens = None
            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states, _ = self.decoder(decoder_input, memory_bank, dec_states, init_tokens, step=step,
                                                  memory_masks=memory_mask, topic_vec=topic_vec, requires_att=True)

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
            topk_beam_index = topk_ids.floor_divide(vocab_size).long()
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

    def _topic_vec_pn(self, batch, topic_info):

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
                topic_vec_cust_summ, topic_vec_cust_noise = topic_vec_cust
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

    def _topic_vec_ge(self, topic_info, agent_mask, cust_mask, idx, vec):

        bsz, max_len = agent_mask.size(0), agent_mask.size(1)
        topic_vec_all, topic_vec_cust, topic_vec_agent = topic_info
        agent_mask, cust_mask = agent_mask.float(), cust_mask.float()

        customer = (topic_vec_cust is not None)
        agent = (topic_vec_agent is not None)

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
                topic_vec_cust_summ, topic_vec_cust_noise = topic_vec_cust
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

        vec = torch.cat([torch.zeros([bsz, vec.size(-1)], device=self.device).unsqueeze(1), vec], 1)
        mapped_vec = torch.cat([vec[i].index_select(0, idx[i]+1).unsqueeze(0) for i in range(bsz)], 0)

        if self.args.split_noise:
            gate_summ = torch.sigmoid(self.topic_gate_linear_summ(torch.cat([mapped_vec, topic_vec[0]], dim=-1)))
            fused_vec_summ = (1-gate_summ) * topic_vec[0] + gate_summ * self.topic_emb_linear_summ(mapped_vec)

            gate_noise = torch.sigmoid(self.topic_gate_linear_noise(torch.cat([mapped_vec, topic_vec[1]], dim=-1)))
            fused_vec_noise = (1-gate_noise) * topic_vec[1] + gate_noise * self.topic_emb_linear_noise(mapped_vec)

            fused_vec = (fused_vec_summ, fused_vec_noise)
        else:
            gate = torch.sigmoid(self.topic_gate_linear(torch.cat([mapped_vec, topic_vec], dim=-1)))
            fused_vec = (1-gate) * topic_vec + gate * self.topic_emb_linear(mapped_vec)
        return fused_vec

    def _add_mask(self, src, mask_src):
        pm_index = torch.empty_like(mask_src).float().uniform_().le(self.args.mask_token_prob)
        ps_index = torch.empty_like(mask_src[:, 0]).float().uniform_().gt(self.args.select_sent_prob)
        pm_index[ps_index] = 0
        # Avoid mask [PAD]
        pm_index[(~mask_src).bool()] = 0
        # Avoid mask [CLS]
        pm_index[:, 0] = 0
        # Avoid mask [SEG]
        pm_index[src == self.seg_token] = 0
        src[pm_index] = self.mask_token
        return src

    def pretrain(self, batch):

        src = batch.src
        mask_src = batch.mask_src
        tgt = batch.tgt
        tgt_labels = batch.tgt_labels
        ex_segs = batch.ex_segs

        tgt_onehot_labels, dup_mask = self._index_2_onehot(tgt_labels, ex_segs)
        setattr(batch, 'pn_tgt', tgt_onehot_labels)
        for i in range(len(ex_segs)):
            if max(tgt_labels[i]) >= ex_segs[i]:
                print(src, tgt, tgt_labels, ex_segs)
                exit()

        # ext module

        # sent encoding
        src_emb = self.embeddings(src)
        sent_hid = self.sent_encoder(src_emb, ~mask_src)[:, 0, :]

        # hierarchical encoding
        sent_list = torch.split(sent_hid, ex_segs)
        sent_input = pad_sequence(sent_list, batch_first=True, padding_value=0.)
        sent_mask_list = [mask_src.new_zeros([length]) for length in ex_segs]
        sent_mask = pad_sequence(sent_mask_list, batch_first=True, padding_value=1)

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
            topic_vec_pn = self._topic_vec_pn(batch, topic_info)
        else:
            topic_loss, topic_vec_pn = None, None

        hier = self.hier_encoder(sent_input, sent_mask)

        pn_decoder_input_list = [hier[i][tgt_labels[i]] for i in range(len(ex_segs))]
        pn_decoder_input = pad_sequence(pn_decoder_input_list, batch_first=True, padding_value=0.)

        pn_decoder_input = torch.cat([self.pn_init_token.unsqueeze(0).expand(len(ex_segs), 1, -1), pn_decoder_input], 1)
        pn_mask_list = [mask_src.new_zeros([len(tgt_label)+1]) for tgt_label in tgt_labels]
        pn_mask = pad_sequence(pn_mask_list, batch_first=True, padding_value=1)


        # pointer net
        if self.training:
            dec_state = self.pn_decoder.init_decoder_state(src, hier)
            pn_decoder_output, _, _ = self.pn_decoder(pn_decoder_input, hier, dec_state,
                                                      memory_masks=sent_mask, tgt_masks=pn_mask,
                                                      topic_vec=topic_vec_pn)
            pn_result = self.pn_generator(hier, pn_decoder_output, sent_mask, pn_mask, dup_mask)
        else:
            pn_result = self._pointer_net_decoding(batch, hier, self.pn_init_token, self.args.max_pos,
                                                   memory_mask=sent_mask, topic_vec=topic_vec_pn, method="max")
            pn_result = [t[0][0].tolist() for t in pn_result]

        # abs module
        # generate abs src
        if self.training:
            src_mapped, segs_mapped, mask_src_mapped, agent_mask, customer_mask, idx_mapped = \
                self._map_src(src, tgt_labels, ex_segs, self.seg_token)
        else:
            src_mapped, segs_mapped, mask_src_mapped, agent_mask, customer_mask, idx_mapped = \
                self._map_src(src, pn_result, ex_segs, self.seg_token)

        # encoding
        if self.args.encoder == "bert":
            top_vec = self.encoder(src_mapped, segs_mapped, mask_src_mapped)
        else:
            src_mapped_emb = self.embeddings(src_mapped)
            top_vec = self.encoder(src_mapped_emb, 1-mask_src_mapped)

        if self.args.topic_model:
            topic_vec_ge = self._topic_vec_ge(topic_info, agent_mask, customer_mask, idx_mapped, hier)
        else:
            topic_vec_ge = None
        # decoding
        if self.training:
            dec_state = self.decoder.init_decoder_state(src, top_vec)
            decode_output, _, _ = self.decoder(tgt[:, :-1], top_vec, dec_state,
                                               memory_masks=~mask_src_mapped,
                                               topic_vec=topic_vec_ge)
            summary = None
        else:
            decode_output = None
            summary = self._fast_translate_batch(batch, top_vec, self.max_length, memory_mask=~mask_src_mapped,
                                                 min_length=2, beam_size=self.beam_size,
                                                 topic_vec=topic_vec_ge)
        return pn_result, decode_output, topic_loss, summary

    def forward(self, batch):

        src = batch.src
        tgt = batch.tgt
        mask_src = batch.mask_src
        ex_segs = batch.ex_segs

        # ext module
        # sent encoding
        src_emb = self.embeddings(src)
        sent_hid = self.sent_encoder(src_emb, ~mask_src)[:, 0, :]

        # hierarchical encoding
        sent_list = torch.split(sent_hid, ex_segs)
        sent_input = pad_sequence(sent_list, batch_first=True, padding_value=0.)
        sent_mask_list = [mask_src.new_zeros([length]) for length in ex_segs]
        sent_mask = pad_sequence(sent_mask_list, batch_first=True, padding_value=1)
        hier = self.hier_encoder(sent_input, sent_mask)

        if self.args.topic_model:
            all_bow, customer_bow, agent_bow = \
                batch.all_bow, batch.customer_bow, batch.agent_bow
            if self.args.split_noise:
                summ_all, summ_customer, summ_agent = \
                    batch.summ_all, batch.summ_customer, batch.summ_agent
            else:
                summ_all, summ_customer, summ_agent = None, None, None
            # with torch.no_grad():
            topic_loss, topic_info = self.topic_model(all_bow, customer_bow, agent_bow,
                                                      summ_all, summ_customer, summ_agent)
            topic_loss = self.args.loss_lambda * topic_loss
            topic_vec_pn = self._topic_vec_pn(batch, topic_info)
        else:
            topic_loss, topic_vec_pn, topic_info = None, None, None

        pn_result = self._pointer_net_decoding(batch, hier, self.pn_init_token,
                                               self.args.max_pos, memory_mask=sent_mask,
                                               topic_vec=topic_vec_pn,
                                               method="sample" if self.training else "max")
        # sample summary generate
        ext_labels = [t[0][0].tolist() for t in pn_result]
        src_mapped, segs_mapped, mask_src_mapped, agent_mask, customer_mask, idx_mapped = \
            self._map_src(src, ext_labels, ex_segs, self.seg_token)

        # Encoding with generated labels.
        if self.args.encoder == "bert":
            top_vec = self.encoder(src_mapped, segs_mapped, mask_src_mapped)
        else:
            src_mapped_emb = self.embeddings(src_mapped)
            top_vec = self.encoder(src_mapped_emb, ~mask_src_mapped)

        if self.args.topic_model:
            topic_vec_ge = self._topic_vec_ge(topic_info, agent_mask, customer_mask, idx_mapped, hier)
        else:
            topic_vec_ge = None

        if self.training:
            dec_state = self.decoder.init_decoder_state(src, top_vec)
            decode_output, _, _ = self.decoder(tgt[:, :-1], top_vec, dec_state,
                                               memory_masks=~mask_src_mapped,
                                               topic_vec=topic_vec_ge)
            with torch.no_grad():

                summary = self._fast_translate_batch(batch, top_vec, self.max_length,
                                                     memory_mask=~mask_src_mapped,
                                                     min_length=2, beam_size=1,
                                                     topic_vec=topic_vec_ge)
                # Get base summaries for reward computation.
                summary_base = self._base_summary_generate(batch, topic_info)

            rl_loss = self._rl(batch, pn_result, summary, summary_base)

        else:
            summary = self._fast_translate_batch(batch, top_vec, self.max_length,
                                                 memory_mask=~mask_src_mapped,
                                                 min_length=2, beam_size=self.args.beam_size,
                                                 topic_vec=topic_vec_ge)
            rl_loss, decode_output = None, None

        return rl_loss, decode_output, topic_loss, summary, ext_labels
