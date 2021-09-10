import os

import torch
from tensorboardX import SummaryWriter

import distributed
from models.reporter import ReportMgr, Statistics
from models.loss import abs_loss
from others.logging import logger


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optims, tokenizer):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]'], 'SEG': tokenizer.vocab['[unused3]'],
               'UNK': tokenizer.vocab['[UNK]']}

    gen_loss = abs_loss(args, model.generator, symbols, tokenizer.vocab, device, train=True)

    trainer = Trainer(args, model, optims, tokenizer, gen_loss,
                      grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optims, tokenizer, abs_loss,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.tokenizer = tokenizer
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.abs_loss = abs_loss

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optims[0]._step + 1
        true_batchs = []
        accum = 0
        tgt_tokens = 0
        src_tokens = 0
        sents = 0
        examples = 0

        train_iter = train_iter_fct()
        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    tgt_tokens += batch.tgt[:, 1:].ne(self.abs_loss.padding_idx).sum().item()
                    src_tokens += batch.src[:, 1:].ne(self.abs_loss.padding_idx).sum().item()
                    sents += batch.src.size(0)
                    examples += batch.tgt.size(0)
                    accum += 1
                    if accum == self.grad_accum_count:
                        if self.n_gpu > 1:
                            tgt_tokens = sum(distributed.all_gather_list(tgt_tokens))
                            src_tokens = sum(distributed.all_gather_list(src_tokens))
                            sents = sum(distributed.all_gather_list(sents))
                            examples = sum(distributed.all_gather_list(examples))

                        normalization = (tgt_tokens, src_tokens, sents, examples)
                        self._gradient_calculation(
                            true_batchs, normalization, total_stats,
                            report_stats, step)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optims[0].learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        src_tokens = 0
                        tgt_tokens = 0
                        sents = 0
                        examples = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)
                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def _gradient_calculation(self, true_batchs, normalization, total_stats,
                              report_stats, step):
        self.model.zero_grad()

        for batch in true_batchs:
            outputs, _, topic_loss = self.model(batch)

            tgt_tokens, src_tokens, sents, examples = normalization

            if self.args.topic_model:
                # Topic Model loss
                topic_stats = Statistics(topic_loss=topic_loss.clone().item() / float(examples))
                topic_loss.div(float(examples)).backward(retain_graph=True)
                total_stats.update(topic_stats)
                report_stats.update(topic_stats)

            # Auto-encoder loss
            abs_stats = self.abs_loss(batch, outputs, self.args.generator_shard_size,
                                      tgt_tokens, retain_graph=False)
            abs_stats.n_docs = len(batch)
            total_stats.update(abs_stats)
            report_stats.update(abs_stats)

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.n_gpu > 1:
            grads = [p.grad.data for p in self.model.parameters()
                     if p.requires_grad
                     and p.grad is not None]
            distributed.all_reduce_and_rescale_tensors(
                grads, float(1))
        for o in self.optims:
            o.step()

    def _save(self, step):
        real_model = self.model

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
