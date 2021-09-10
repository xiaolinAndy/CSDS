export CUDA_VISIBLE_DEVICES=0

python preprocess.py -log_file logs/log

# bert abs final
mode=final
python train.py -task abs -mode train -bert_data_path data/bert/both/${mode} -dec_dropout 0.2  -model_path output/bert_abs_${mode} -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 2  -log_file logs/bert_abs_train_${mode}.log -finetune_bert True
python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/bert/both/${mode} -log_file logs/bert_abs_val_${mode}.log -model_path output/bert_abs_${mode} -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_abs_val_${mode}.txt -temp_dir temp/ -test_all=True
python train.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/bert/both/${mode} -log_file logs/bert_abs_test_${mode}.log -test_from output/bert_abs_${mode}/model_step_1600.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_abs_${mode}.txt -temp_dir temp/


##bert abs user
#mode=user
#python train.py -task abs -mode train -bert_data_path data/bert/both/${mode} -dec_dropout 0.2  -model_path output/bert_abs_${mode} -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 2  -log_file logs/bert_abs_train_${mode}.log -finetune_bert True
#python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/bert/both/${mode} -log_file logs/bert_abs_val_${mode}.log -model_path output/bert_abs_${mode} -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_abs_val_${mode}.txt -temp_dir temp/ -test_all=True
#python train.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/bert/both/${mode} -log_file logs/bert_abs_test_${mode}.log -test_from output/bert_abs_${mode}/model_step_xxx.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_abs_${mode}.txt -temp_dir temp/

#
##bert abs agent
#model=agent
#python train.py -task abs -mode train -bert_data_path data/bert/both/${mode} -dec_dropout 0.2  -model_path output/bert_abs_${mode} -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 2  -log_file logs/bert_abs_train_${mode}.log -finetune_bert True
#python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/bert/both/${mode} -log_file logs/bert_abs_val_${mode}.log -model_path output/bert_abs_${mode} -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_abs_val_${mode}.txt -temp_dir temp/ -test_all=True
#python train.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/bert/both/${mode} -log_file logs/bert_abs_test_${mode}.log -test_from output/bert_abs_${mode}/model_step_xxx.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_abs_${mode}.txt -temp_dir temp/

#
##bert ext final
#mode=final
#python train.py -task ext -mode train -bert_data_path data/bert/both/${mode} -ext_dropout 0.1 -model_path output/bert_ext_${mode}/ -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -accum_count 15 -log_file logs/ext_bert -use_interval true -warmup_steps 1000 -max_pos 512
#python train.py -task ext -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/bert/both/${mode} -log_file logs/bert_ext_val_${mode}.log -model_path output/bert_ext_${mode} -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_ext_val_${mode}.txt -temp_dir temp/ -test_all=True
#python train.py -task ext -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/bert/both/${mode} -log_file logs/bert_ext_test_${mode}.log -test_from output/bert_ext_${mode}/model_step_xxx.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_ext_${mode}.txt -temp_dir temp/ -ratio 100

#
##bert ext user
#mode=user
#python train.py -task ext -mode train -bert_data_path data/bert/both/${mode} -ext_dropout 0.1 -model_path output/bert_ext_${mode}/ -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -accum_count 15 -log_file logs/ext_bert -use_interval true -warmup_steps 1000 -max_pos 512
#python train.py -task ext -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/bert/both/${mode} -log_file logs/bert_ext_val_${mode}.log -model_path output/bert_ext_${mode} -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_ext_val_${mode}.txt -temp_dir temp/ -test_all=True
#python train.py -task ext -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/bert/both/${mode} -log_file logs/bert_ext_test_${mode}.log -test_from output/bert_ext_${mode}/model_step_xxx.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_ext_${mode}.txt -temp_dir temp/ -ratio 100

#
##bert ext agent
#mode=agent
#python train.py -task ext -mode train -bert_data_path data/bert/both/${mode} -ext_dropout 0.1 -model_path output/bert_ext_${mode}/ -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -accum_count 15 -log_file logs/ext_bert -use_interval true -warmup_steps 1000 -max_pos 512
#python train.py -task ext -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/bert/both/${mode} -log_file logs/bert_ext_val_${mode}.log -model_path output/bert_ext_${mode} -sep_optim true -use_interval true -visible_gpus 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_ext_val_${mode}.txt -temp_dir temp/ -test_all=True
#python train.py -task ext -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/bert/both/${mode} -log_file logs/bert_ext_test_${mode}.log -test_from output/bert_ext_${mode}/model_step_xxx.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 30 -result_path logs/bert_ext_${mode}.txt -temp_dir temp/ -ratio 100
