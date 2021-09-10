export DATA=dataset/
export CUDA_VISIBLE_DEVICES=0

#cp ../PGN/data_utils/embeddings/dialogue_embed_word data_utils/embeddings/dialogue_embed_word
# final
mode=final
rm -rf dataset/train
rm -rf dataset/val
rm -rf dataset/test
rm -rf saved/complete/${mode}/
mkdir saved/complete/${mode}/
python make_extraction_labels.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both --complete
python train_abstractor.py --path=saved/complete/${mode}/abs/
python train_extractor_ml.py --path=saved/complete/${mode}/ext/
python train_full_rl.py --path=saved/complete/${mode}/full --abs_dir=saved/complete/${mode}/abs --ext_dir=saved/complete/${mode}/ext
python make_eval_references.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both --complete
python decode_full_model.py --path=saved/complete/${mode}/decode --model_dir=saved/complete/${mode}/full --beam=5 --test


# user
#mode=user
#rm -rf dataset/train
#rm -rf dataset/val
#rm -rf dataset/test
#rm -rf saved/complete/${mode}/
#mkdir saved/complete/${mode}/
#python make_extraction_labels.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both --complete
#python train_abstractor.py --path=saved/complete/${mode}/abs/
#python train_extractor_ml.py --path=saved/complete/${mode}/ext/
#python train_full_rl.py --path=saved/complete/${mode}/full --abs_dir=saved/complete/${mode}/abs --ext_dir=saved/complete/${mode}/ext
#python make_eval_references.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both --complete
#python decode_full_model.py --path=saved/complete/${mode}/decode --model_dir=saved/complete/${mode}/full --beam=5 --test

#
## agent
#mode=agent
#rm -rf dataset/train
#rm -rf dataset/val
#rm -rf dataset/test
#rm -rf saved/complete/${mode}/
#mkdir saved/complete/${mode}/
#python make_extraction_labels.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both --complete
#python train_abstractor.py --path=saved/complete/${mode}/abs/
#python train_extractor_ml.py --path=saved/complete/${mode}/ext/
#python train_full_rl.py --path=saved/complete/${mode}/full --abs_dir=saved/complete/${mode}/abs --ext_dir=saved/complete/${mode}/ext
#python make_eval_references.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both --complete
#python decode_full_model.py --path=saved/complete/${mode}/decode --model_dir=saved/complete/${mode}/full --beam=5 --test
