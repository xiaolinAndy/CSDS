import json
import os
from utils import config
from train.trainer import PGNTrainer

def get_best_model_pth(dir_path):
    min_loss = 1e9
    best_model = ''
    for name in os.listdir(dir_path):
        split_name = name.split('_')
        if float(split_name[0]) < min_loss:
            min_loss = float(split_name[0])
            best_model = name
    return best_model


if __name__ == '__main__':
    args = config.args
    if args.do_train:
        print('start training-------')
        print(args)
        trainer = PGNTrainer(args.plot_path, args.gpu_id)
        trainer.trainIters(args.epochs)
    elif args.do_ft:
        print('start finetuning-------')
        print(args)
        trainer = PGNTrainer(args.plot_path, args.gpu_id)
        best_model = get_best_model_pth(os.path.join(args.save_path, 'checkpoints'))
        trainer.trainIters(args.epochs, model_file_path=os.path.join(os.path.join(args.save_path, 'checkpoints'), best_model))
    elif args.do_eval:
        print('start eval-------')
        print(args)
        trainer = PGNTrainer(args.plot_path, args.gpu_id)
        if args.best_model_pth is None:
            best_model = get_best_model_pth(os.path.join(args.save_path, 'checkpoints'))
            trainer.eval(os.path.join(os.path.join(args.save_path, 'checkpoints'), best_model))
        else:
            trainer.eval(args.best_model_pth)

