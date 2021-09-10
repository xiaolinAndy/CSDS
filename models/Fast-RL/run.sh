export DATA=dataset/
export CUDA_VISIBLE_DEVICES=0

#cp ../PGN/data_utils/embeddings/dialogue_embed_word data_utils/embeddings/dialogue_embed_word
# final
mode=final
rm -rf dataset/train
rm -rf dataset/val
rm -rf dataset/test
rm -rf saved/${mode}/
mkdir saved/${mode}/
python make_extraction_labels.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both
python train_abstractor.py --path=saved/${mode}/abs/
python train_extractor_ml.py --path=saved/${mode}/ext/
python train_full_rl.py --path=saved/${mode}/full --abs_dir=saved/${mode}/abs --ext_dir=saved/${mode}/ext
python make_eval_references.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both
python decode_full_model.py --path=saved/${mode}/decode --model_dir=saved/${mode}/full --beam=5 --test


## user
#rm -rf dataset/train
#rm -rf dataset/val
#rm -rf dataset/test
#mode=user
#mkdir saved/${mode}/
#python make_extraction_labels.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both
#python train_abstractor.py --path=saved/${mode}/abs/
#python train_extractor_ml.py --path=saved/${mode}/ext/
#python train_full_rl.py --path=saved/${mode}/full --abs_dir=saved/${mode}/abs --ext_dir=saved/${mode}/ext
#python make_eval_references.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both
#python decode_full_model.py --path=saved/${mode}/decode --model_dir=saved/${mode}/full --beam=5 --test


# agent
#rm -rf dataset/train
#rm -rf dataset/val
#rm -rf dataset/test
#mode=agent
#mkdir saved/${mode}/
#python make_extraction_labels.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both
#python train_abstractor.py --path=saved/${mode}/abs/
#python train_extractor_ml.py --path=saved/${mode}/ext/
#python train_full_rl.py --path=saved/${mode}/full --abs_dir=saved/${mode}/abs --ext_dir=saved/${mode}/ext
#python make_eval_references.py --split_mode=period --turn_mode=multi --sum_mode=${mode} --context_mode=both
#python decode_full_model.py --path=saved/${mode}/decode --model_dir=saved/${mode}/full --beam=5 --test
