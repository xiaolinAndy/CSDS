import random
import numpy as np
import torch
import os
import time
import torch.nn.functional as F
import files2rouge
import re

from utils import config
from model import pgn
from data_utils.batch import get_train_dataloader, get_val_dataloader
from train.train_utils import get_input_from_batch, get_output_from_batch
from data_utils.tokenizer import Tokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad, Adam
from utils.metric import cal_mul_sums_n, cal_mul_sums_l

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)

def change_word2id(ref, pred):
    ref_id, pred_id = [], []
    tmp_dict = {}
    new_index = 0
    words = list(ref)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            ref_id.append(str(new_index))
            new_index += 1
        else:
            ref_id.append(str(tmp_dict[w]))
    words = list(pred)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            pred_id.append(str(new_index))
            new_index += 1
        else:
            pred_id.append(str(tmp_dict[w]))
    return ' '.join(ref_id), ' '.join(pred_id)


class BeamSearch(object):
    def __init__(self, model, vocab, dataloader, device, save, args):
        model.eval()
        self.vocab = vocab
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.save_dir = args.save_path
        self.save = save

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        refs, preds = [], []
        start_time = time.time()
        for batch in self.dataloader:
            # Run beam search to get best Hypothesis
            art_oovs = batch[0][3][0]
            _, _, _, _, _,  original_abstract_sents= \
                get_output_from_batch(batch, self.device, self.vocab)
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = self.vocab.outputids2words(output_ids, art_oovs)

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index('<END>')
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            # calulate based on character
            refs.append(''.join(original_abstract_sents[0].split()))
            preds.append(''.join(decoded_words))

        print('ref: ', refs[0])
        print('pred: ', preds[0])
        print('time: ', time.time()-start_time)
        # using files2rouge as calculating method, chinese
        ref_ids, pred_ids = [], []
        if self.save:
            with open(self.save_dir + 'refs.txt', 'w') as f:
                for ref in refs:
                    f.write(ref + '\n')
            with open(self.save_dir + 'preds.txt', 'w') as f:
                for pred in preds:
                    f.write(pred + '\n')
        for ref, pred in zip(refs, preds):
            ref_id, pred_id = change_word2id(ref, pred)
            ref_ids.append(ref_id)
            pred_ids.append(pred_id)

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        with open(self.save_dir + 'ref_ids.txt', 'w') as f:
            for ref in ref_ids:
                f.write(ref + '\n')
        with open(self.save_dir + 'pred_ids.txt', 'w') as f:
            for pred in pred_ids:
                f.write(pred + '\n')
        files2rouge.run(self.save_dir + 'pred_ids.txt', self.save_dir + 'ref_ids.txt')
        if self.save:
            os.system('files2rouge %s/ref_ids.txt  %s/pred_ids.txt -s %s/rouge_score.txt' % (self.save_dir, self.save_dir, self.save_dir))

    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = get_input_from_batch(batch, self.device, self.vocab)
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.token2idx('<START>')],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.args.beam_size)]
        results = []
        steps = 0
        while steps < config.args.max_dec_steps and len(results) < config.args.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.vocab_size else self.vocab.token2idx('[UNK]') \
                             for t in latest_tokens]
            y_t_1 = torch.LongTensor(latest_tokens)
            y_t_1 = y_t_1.to(self.device)
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)


            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.args.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.args.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.token2idx('<END>'):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.args.beam_size or len(results) == config.args.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

class PGNTrainer(object):
    def __init__(self, plot_path, gpu_id):
        self.args = config.args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        self.device = torch.device('cuda', gpu_id) if gpu_id >= 0 else torch.device('cpu')
        self.gpu = True if gpu_id >= 0 else False
        self.model = None
        self.train_dataloader, self.train_data, self.vocab = get_train_dataloader(self.args.train_pth, self.args.max_seq_len,
                                                                      self.args.batch_size)
        self.val_dataloader, self.val_data = get_val_dataloader(self.args.val_pth, self.vocab, self.args.max_seq_len,
                                                                self.args.batch_size)
        self.val_decode_dataloader, _ = get_val_dataloader(self.args.val_pth, self.vocab, self.args.max_seq_len,
                                                                self.args.batch_size, mode='decode', beam_size=self.args.beam_size)
        self.test_dataloader, self.test_data = get_val_dataloader(self.args.test_pth, self.vocab, self.args.max_seq_len,
                                                                  self.args.batch_size, mode='decode', beam_size=self.args.beam_size)
        self.optimizer = None
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        if not os.path.exists(os.path.join(self.args.save_path, 'checkpoints')):
            os.makedirs(os.path.join(self.args.save_path, 'checkpoints'))
        self.summary_writer = SummaryWriter(plot_path)
        self.min_val_loss = 10
        self.embedding = torch.tensor(self.vocab.embedding, dtype=torch.float).to(self.device)


    def __init_model(self):
        self.model.to(self.device)

    def new_model(self):
        self.model = pgn.PGNModel()
        self.__init_model()

    # TODO: change load model
    def load_model(self, path_model):
        self.model = BertForSequenceClassification(config)
        self.model.load_state_dict(torch.load(path_model))
        self.__init_model()


    def save_model(self, running_avg_loss, iter, val_loss):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(os.path.join(self.args.save_path, 'checkpoints'), '%.3f_model_%d' % (val_loss, iter))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = pgn.PGNModel(model_file_path, device=self.device, embedding=self.embedding)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        #self.optimizer = Adam(params)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if self.gpu:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, self.device, self.vocab)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch, _ = \
            get_output_from_batch(batch, self.device, self.vocab)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, self.args.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()
        # if torch.isnan(loss).any():
        #     exit()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def test_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, self.device, self.vocab)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch, _ = \
            get_output_from_batch(batch, self.device, self.vocab)

        with torch.no_grad():
            encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
            s_t_1 = self.model.reduce_state(encoder_hidden)

            step_losses = []
            for di in range(min(max_dec_len, self.args.max_dec_steps)):
                y_t_1 = dec_batch[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                               encoder_outputs,
                                                                                               encoder_feature,
                                                                                               enc_padding_mask, c_t_1,
                                                                                               extra_zeros,
                                                                                               enc_batch_extend_vocab,
                                                                                               coverage, di)
                target = target_batch[:, di]
                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
                step_loss = -torch.log(gold_probs + config.eps)
                if config.is_coverage:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                    coverage = next_coverage

                step_mask = dec_padding_mask[:, di]
                step_loss = step_loss * step_mask
                step_losses.append(step_loss)

            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / dec_lens_var
            loss = torch.mean(batch_avg_loss)
            return loss.item()

    def calc_running_avg_loss(self, loss, running_avg_loss, summary_writer, step, decay=0.99):
        if running_avg_loss == 0:  # on the first iteration just take the loss
            running_avg_loss = loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
        running_avg_loss = min(running_avg_loss, 12)  # clip
        tag_name = 'running_avg_loss/decay=%f' % (decay)
        summary_writer.add_scalar(tag_name, running_avg_loss, step)
        return running_avg_loss

    def trainIters(self, epochs, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        #losses, save = self.validate(iter)
        #self.test(save)
        #self.save_model(running_avg_loss, iter, losses)
        for e in range(epochs):
            print("Epoch {}".format(e))
            for step, batch in enumerate(self.train_dataloader):
                self.model.train()
                loss = self.train_one_batch(batch)

                running_avg_loss = self.calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
                iter += 1
                if iter % self.args.log_freq == 0:
                    self.summary_writer.flush()
                print_interval = self.args.log_freq
                if iter % print_interval == 0:
                    print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                               time.time() - start, loss))
                    start = time.time()
                if iter % self.args.val_freq == 0:
                    losses, save = self.validate(iter)
                    self.test(save)
                    self.save_model(running_avg_loss, iter, losses)
        print('min loss: ', self.min_val_loss)
        print('best step: ', self.best_step)


    def validate(self, train_step):
        print('Begin Validating-------------------')
        losses = 0.
        batch_count = 0
        for step, batch in enumerate(self.val_dataloader):
            loss = self.test_one_batch(batch)
            losses += loss
            batch_count += 1
        losses /= batch_count
        print('validation loss: %f' % losses)
        if losses < self.min_val_loss:
            self.min_val_loss = losses
            self.best_step = train_step
            save = True
        else:
            save = False
        return losses, save

    def test(self, save):
        print('Begin Testing-------------------')
        beam_Search_processor = BeamSearch(self.model, self.vocab, self.test_dataloader, self.device, save, self.args)
        beam_Search_processor.decode()


