#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import torch
import distributed

from pytorch_transformers import BertTokenizer
from models import data_loader
from models.data_loader import load_dataset
from models.optimizers import build_optim, build_optim_bert, build_optim_other, build_optim_topic
from models.rl_model import Model as Summarizer
from models.rl_predictor import build_predictor
from models.rl_model_trainer import build_trainer
from others.logging import logger, init_logger
from others.utils import rouge_results_to_str, test_bleu, test_length
from rouge import Rouge, FilesRouge

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def baseline(args, cal_lead=False, cal_oracle=False):
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, args.test_batch_ex_size, 'cpu',
                                       shuffle=False, is_test=True)

    if cal_lead:
        mode = "lead"
    else:
        mode = "oracle"

    rouge = Rouge()
    pred_path = '%s.%s.pred' % (args.result_path, mode)
    gold_path = '%s.%s.gold' % (args.result_path, mode)
    save_pred = open(pred_path, 'w', encoding='utf-8')
    save_gold = open(gold_path, 'w', encoding='utf-8')

    with torch.no_grad():
        count = 0
        for batch in test_iter:
            summaries = batch.tgt_txt
            origin_sents = batch.original_str
            ex_segs = batch.ex_segs
            ex_segs = [sum(ex_segs[:i]) for i in range(len(ex_segs)+1)]

            for idx in range(len(summaries)):
                summary = " ".join(summaries[idx][1:-1]).replace("\n", "")
                txt = origin_sents[ex_segs[idx]:ex_segs[idx+1]]
                if cal_oracle:
                    selected = []
                    max_rouge = 0.
                    while True:
                        cur_max_rouge = max_rouge
                        cur_id = -1
                        for i in range(len(txt)):
                            if (i in selected):
                                continue
                            c = selected + [i]
                            temp_txt = " ".join([txt[j][9:] for j in c])
                            if len(temp_txt.split()) > args.ex_max_token_num:
                                continue
                            rouge_score = rouge.get_scores(temp_txt, summary)
                            rouge_1 = rouge_score[0]["rouge-1"]["f"]
                            rouge_l = rouge_score[0]["rouge-l"]["f"]
                            rouge_score = rouge_1 + rouge_l
                            if rouge_score > cur_max_rouge:
                                cur_max_rouge = rouge_score
                                cur_id = i
                        if (cur_id == -1):
                            break
                        selected.append(cur_id)
                        max_rouge = cur_max_rouge
                    pred_txt = " ".join([txt[j][9:] for j in selected])
                else:
                    k = min(args.ex_max_k, len(txt))
                    pred_txt = " ".join(txt[:k]).replace("\n", "")
                save_gold.write(summary + "\n")
                save_pred.write(pred_txt + "\n")
                count += 1
                if count % 10 == 0:
                    print(count)
    save_gold.flush()
    save_pred.flush()
    save_gold.close()
    save_pred.close()

    length = test_length(pred_path, gold_path)
    bleu = test_bleu(pred_path, gold_path)
    file_rouge = FilesRouge(hyp_path=pred_path, ref_path=gold_path)
    pred_rouges = file_rouge.get_scores(avg=True)
    logger.info('Length ratio:\n%s' % str(length))
    logger.info('Bleu:\n%s' % str(bleu*100))
    logger.info('Rouges:\n%s' % rouge_results_to_str(pred_rouges))


def validate(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        best_dev_steps = -1
        best_dev_results = (0, 0)
        best_test_results = (0, 0)
        patient = 100
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            logger.info("Step %d: processing %s" % (i, cp))
            rouge_dev = validate_(args, device_id, cp, step)
            rouge_test = test(args, device_id, cp, step)
            if (rouge_dev[0] + rouge_dev[1]) > (best_dev_results[0] + best_dev_results[1]):
                best_dev_results = rouge_dev
                best_test_results = rouge_test
                best_dev_steps = step
                patient = 100
            else:
                patient -= 1
            logger.info("Current step: %d" % step)
            logger.info("Dev results: ROUGE-1-l: %f, %f" % (rouge_dev[0], rouge_dev[1]))
            logger.info("Test results: ROUGE-1-l: %f, %f" % (rouge_test[0], rouge_test[1]))
            logger.info("Best step: %d" % best_dev_steps)
            logger.info("Best dev results: ROUGE-1-l: %f, %f" % (best_dev_results[0], best_dev_results[1]))
            logger.info("Best test results: ROUGE-1-l: %f, %f\n\n" % (best_test_results[0], best_test_results[1]))

            if patient == 0:
                break

    else:
        best_dev_results = (0, 0)
        best_test_results = (0, 0)
        best_dev_steps = -1
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    rouge_dev = validate_(args, device_id, cp, step)
                    rouge_test = test(args, device_id, cp, step)
                    if (rouge_dev[0] + rouge_dev[1]) > (best_dev_results[0] + best_dev_results[1]):
                        best_dev_results = rouge_dev
                        best_test_results = rouge_test
                        best_dev_steps = step

                    logger.info("Current step: %d" % step)
                    logger.info("Dev results: ROUGE-1-l: %f, %f" % (rouge_dev[0], rouge_dev[1]))
                    logger.info("Test results: ROUGE-1-l: %f, %f" % (rouge_test[0], rouge_test[1]))
                    logger.info("Best step: %d" % best_dev_steps)
                    logger.info("Best dev results: ROUGE-1-l: %f, %f" % (best_dev_results[0], best_dev_results[1]))
                    logger.info("Best test results: ROUGE-1-l: %f, %f\n\n" % (best_test_results[0], best_test_results[1]))

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate_(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'val', shuffle=False),
                                        args.test_batch_size, args.test_batch_ex_size, device,
                                        shuffle=False, is_test=True)

    predictor = build_predictor(args, tokenizer, model, logger)
    rouge = predictor.validate(valid_iter, step)
    return rouge


def test(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, args.test_batch_ex_size, device,
                                       shuffle=False, is_test=True)

    predictor = build_predictor(args, tokenizer, model, logger)
    rouge = predictor.validate(test_iter, step)
    return rouge


def test_text(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, args.test_batch_ex_size, device,
                                       shuffle=False, is_test=True)
    predictor = build_predictor(args, tokenizer, model, logger)
    predictor.translate(test_iter, step)


def train(args, device_id):
    if (args.world_size > 1):
        train_multi(args)
    else:
        train_single(args, device_id)


def train_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True),
                                      args.batch_size, args.batch_ex_size, device,
                                      shuffle=True, is_test=False)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = Summarizer(args, device, tokenizer.vocab, checkpoint)

    if args.train_from_ignore_optim:
        checkpoint = None
    if args.sep_optim:
        if args.encoder == 'bert':
            optim_bert = build_optim_bert(args, model, checkpoint)
            optim_other = build_optim_other(args, model, checkpoint)
            if args.topic_model:
                optim_topic = build_optim_topic(args, model, checkpoint)
                optim = [optim_bert, optim_other, optim_topic]
            else:
                optim = [optim_bert, optim_other]
        else:
            optim_other = build_optim_other(args, model, checkpoint)
            if args.topic_model:
                optim_topic = build_optim_topic(args, model, checkpoint)
                optim = [optim_other, optim_topic]
            else:
                optim = [optim_other]
    else:
        optim = [build_optim(args, model, checkpoint, args.warmup)]

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim, tokenizer)

    if args.pretrain:
        trainer.train(train_iter_fct, args.pretrain_steps)
    else:
        trainer.train(train_iter_fct, args.train_steps)
