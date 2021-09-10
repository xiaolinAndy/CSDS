mkdir json_data/final/
mkdir json_data/user/
mkdir json_data/agent/
python ./src/prepro/convert_json_format.py
python ./src/train_emb.py -data_path json_data/final -emb_size 100 -emb_path pretrain_emb/word2vec
mkdir bert_data/final/
mkdir bert_data/user/
mkdir bert_data/agent/
python ./src/preprocess.py -raw_path json_data/final -save_path bert_data/final/ -bert_dir bert/chinese_bert -log_file logs/preprocess.log -emb_path pretrain_emb/word2vec -tokenize -truncated -add_ex_label
python ./src/preprocess.py -raw_path json_data/user -save_path bert_data/user/ -bert_dir bert/chinese_bert -log_file logs/preprocess.log -emb_path pretrain_emb/word2vec -tokenize -truncated -add_ex_label
python ./src/preprocess.py -raw_path json_data/agent -save_path bert_data/agent/ -bert_dir bert/chinese_bert -log_file logs/preprocess.log -emb_path pretrain_emb/word2vec -tokenize -truncated -add_ex_label

# topic

# final
mode=final
python ./src/train.py -data_path bert_data/${mode}/csds -bert_dir bert/chinese_bert -log_file logs/pipeline.train.${mode}.log -sep_optim -pretrain -model_path models/pipeline_${mode} -visible_gpus 2 -topic_model -split_noise -idf_info_path bert_data/${mode}/idf_info.pt
python ./src/train.py -data_path bert_data/${mode}/csds -bert_dir bert/chinese_bert -log_file logs/rl.train.${mode}.log -model_path models/rl_${mode} -train_from models/pipeline_${mode}/model_step_80000.pt -train_from_ignore_optim -lr 0.00001 -save_checkpoint_steps 1000 -train_steps 10000 -visible_gpus 2 -topic_model -split_noise -idf_info_path bert_data/${mode}/idf_info.pt
python ./src/train.py -mode validate -data_path bert_data/${mode}/csds -bert_dir bert/chinese_bert -log_file logs/rl.val.${mode}.log -alpha 0.95 -model_path models/rl_${mode} -result_path results/val_${mode} -visible_gpus 2 -test_all -topic_model -split_noise -idf_info_path bert_data/${mode}/idf_info.pt
python ./src/train.py -mode test -data_path bert_data/${mode}/csds -bert_dir bert/chinese_bert -test_from models/rl_${mode}/model_step_0.pt -log_file logs/rl.test.${mode}.log -alpha 0.95 -result_path results/test_${mode} -visible_gpus 2 -topic_model -split_noise -idf_info_path bert_data/${mode}/idf_info.pt
#cp results/test_${mode}.0.pred_test ../../results/${mode}/bert_rl_mod_preds.txt
#cp results/test_${mode}.0.gold_test ../../results/${mode}/bert_rl_mod_refs.txt

# user
# python ./src/train.py -data_path bert_data/user/csds -bert_dir bert/chinese_bert -log_file logs/pipeline.train.user.log -sep_optim -pretrain -model_path models/pipeline_user -visible_gpus 2 -topic_model -split_noise
# python ./src/train.py -data_path bert_data/user/csds -bert_dir bert/chinese_bert -log_file logs/rl.train.user.log -model_path models/rl_user -train_from models/pipeline_user/model_step_80000.pt -train_from_ignore_optim -lr 0.00001 -save_checkpoint_steps 1000 -train_steps 10000 -visible_gpus 2 -topic_model -split_noise
# python ./src/train.py -mode validate -data_path bert_data/user/csds -bert_dir bert/chinese_bert -log_file logs/rl.val.user.log -alpha 0.95 -model_path models/rl_user -result_path results/val -visible_gpus 2 -test_all -topic_model -split_noise
# python ./src/train.py -mode test -data_path bert_data/user/csds -bert_dir bert/chinese_bert -test_from models/rl_user/model_step_1000.pt -log_file logs/rl.test.user.log -alpha 0.95 -result_path results/test_user -visible_gpus 2 -topic_model -split_noise
# cp results/test_user.1000.pred_test ../../results/user/bert_rl_mod_preds.txt
# cp results/test_user.1000.gold_test ../../results/user/bert_rl_mod_refs.txt

# agent
# python ./src/train.py -data_path bert_data/agent/csds -bert_dir bert/chinese_bert -log_file logs/pipeline.train.agent.log -sep_optim -pretrain -model_path models/pipeline_agent -visible_gpus 2 -topic_model -split_noise
# python ./src/train.py -data_path bert_data/agent/csds -bert_dir bert/chinese_bert -log_file logs/rl.train.agent.log -model_path models/rl_agent -train_from models/pipeline_agent/model_step_80000.pt -train_from_ignore_optim -lr 0.00001 -save_checkpoint_steps 1000 -train_steps 10000 -visible_gpus 2 -topic_model -split_noise
# python ./src/train.py -mode validate -data_path bert_data/agent/csds -bert_dir bert/chinese_bert -log_file logs/rl.val.agent.log -alpha 0.95 -model_path models/rl_agent -result_path results/val -visible_gpus 2 -test_all -topic_model -split_noise
# python ./src/train.py -mode test -data_path bert_data/agent/csds -bert_dir bert/chinese_bert -test_from models/rl_agent/model_step_1000.pt -log_file logs/rl.test.agent.log -alpha 0.95 -result_path results/test_agent -visible_gpus 2 -topic_model -split_noise
# cp results/test_agent.1000.pred_test ../../results/agent/bert_rl_mod_preds.txt
# cp results/test_agent.1000.gold_test ../../results/agent/bert_rl_mod_refs.txt
