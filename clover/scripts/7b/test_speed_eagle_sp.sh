export CUDA_VISIBLE_DEVICES=0

python -m clover.evaluation.gen_ea_answer_vicuna_sp \
		--ea-model-path /data_train/infra/glj/models/eagle/vicuna-7b-v1.5/state_20 \
		--base-model-path /data_train/infra/glj/models/vicuna-7b-v1.5 \
		--temperature 0
