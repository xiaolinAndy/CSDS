export CUDA_VISIBLE_DEVICES=0
# final
python main.py --do_train --sum_mode=final --context_mode=both --gpu_id=0 --epochs=30 --save_path=output/final/ #--new_vocab
python main.py --do_ft --sum_mode=final --context_mode=both --gpu_id=0 --coverage=True --epochs=10 --save_path=output/final/  --val_freq=1000
python main.py --do_eval --sum_mode=final --context_mode=both --gpu_id=0 --coverage=True --best_model_pth=xxx

# user
#python main.py --do_train --sum_mode=user --context_mode=both --gpu_id=0 --epochs=30 --save_path=output/user/
#python main.py --do_ft --sum_mode=user --context_mode=both --gpu_id=0 --coverage=True --epochs=10 --save_path=output/user/
#python main.py --do_eval --sum_mode=user --context_mode=both --gpu_id=0 --coverage=True --best_model_pth=xxx

# agent
#python main.py --do_train --sum_mode=agent --context_mode=both --gpu_id=0 --epochs=30 --save_path=output/agent/
#python main.py --do_ft --sum_mode=agent --context_mode=both --gpu_id=0 --coverage=True --epochs=10 --save_path=output/agent/ --val_freq=1000
#python main.py --do_eval --sum_mode=agent --context_mode=both --gpu_id=0 --coverage=True --best_model_pth=xxx
