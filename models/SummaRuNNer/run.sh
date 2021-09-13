export CUDA_VISIBLE_DEVICES=0
#cp ../../formal_data/train.json data/train.json
#cp ../../formal_data/val.json data/val.json
#cp ../../formal_data/test.json data/test.json
#cp ../PGN/data_utils/embeddings/dialogue_embed_word data/embeddings/dialogue_embed_word

# final
python main.py \
-device 0 \
-batch_size 32 \
-model RNN_RNN \
-seed 2020 \
-save_path checkpoints/RNN_RNN_seed_2020_final.pt \
-sum_mode final \
-context_mode both \
-vocab_path data/embeddings/dialogue_embed_word

python main.py \
-device 0 \
-batch_size 1 \
-test \
-load_dir checkpoints/RNN_RNN_seed_2020_final.pt \
-sum_mode final \
-context_mode both \
-vocab_path data/embeddings/dialogue_embed_word \
-max_sum 84 \
-output_dir outputs/final/
cp outputs/final/hyp.txt ../../results/final/summarunner_preds.txt
cp outputs/final/ref.txt ../../results/final/summarunner_refs.txt

#user
#python main.py \
#-device 0 \
#-batch_size 32 \
#-model RNN_RNN \
#-seed 2020 \
#-save_path checkpoints/RNN_RNN_seed_2020_user.pt \
#-sum_mode user \
#-context_mode both \
#-vocab_path data/embeddings/dialogue_embed_word
#
#python main.py \
#-device 0 \
#-batch_size 1 \
#-test \
#-load_dir checkpoints/RNN_RNN_seed_2020_user.pt \
#-sum_mode user \
#-context_mode both \
#-vocab_path data/embeddings/dialogue_embed_word \
#-max_sum 38 \
#-output_dir outputs/user/
#cp outputs/user/hyp.txt ../../results/user/summarunner_preds.txt
#cp outputs/user/ref.txt ../../results/user/summarunner_refs.txt
#
##agent
#python main.py \
#-device 0 \
#-batch_size 32 \
#-model RNN_RNN \
#-seed 2020 \
#-save_path checkpoints/RNN_RNN_seed_2020_agent.pt \
#-sum_mode agent \
#-context_mode both \
#-vocab_path data/embeddings/dialogue_embed_word
#
#python main.py \
#-device 0 \
#-batch_size 1 \
#-test \
#-load_dir checkpoints/RNN_RNN_seed_2020_agent.pt \
#-sum_mode agent \
#-context_mode both \
#-vocab_path data/embeddings/dialogue_embed_word \
#-max_sum 49 \
#-output_dir outputs/agent/
#cp outputs/agent/hyp.txt ../../results/agent/summarunner_preds.txt
#cp outputs/agent/ref.txt ../../results/agent/summarunner_refs.txt


