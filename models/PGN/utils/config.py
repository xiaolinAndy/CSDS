import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--save_path", type=str, default='output/')
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--plot_path", type=str, default='output/plot')
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--load_frompretrain", type=str, default="None")
parser.add_argument("--model_state_path", type=str, default="None")
parser.add_argument("--model_config_path", type=str, default="None")
parser.add_argument("--log_freq", type=int, default=200)
parser.add_argument("--val_freq", type=int, default=2000)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--do_ft", action='store_true')
parser.add_argument("--vocab_path", type=str, default="data_utils/embeddings/dialogue_embed_word")
parser.add_argument("--vocab_dim", type=int, default=200)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--pointer_gen", type=bool, default=True)
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--split_word", action='store_true', default=True)
parser.add_argument("--new_vocab", action='store_true')
parser.add_argument("--sum_mode", type=str, default='final')
parser.add_argument("--context_mode", type=str, default='both')
parser.add_argument("--augment", type=bool, default=False)
parser.add_argument("--coverage", type=bool, default=False)
parser.add_argument("--complete", action='store_true')

args = parser.parse_args()

args.train_pth = args.data_path + 'train.json'
args.val_pth = args.data_path + 'val.json'
args.test_pth = args.data_path + 'test.json'

# 800 for char, 500 for word
args.max_seq_len = 500
args.max_dec_steps = 100
args.seed = 2020

# PGN paras
hidden_dim= 256
emb_dim= 200
min_dec_steps=15
lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0
is_coverage = args.coverage
pointer_gen = True
cov_loss_wt = 1.0
eps = 1e-12
max_iterations = 20000
use_gpu=True
lr_coverage=0.15

