export CUDA_VISIBLE_DEVICES=0

python -m clover.evaluation.gen_baseline_answer_vicuna \
		--ea-model-path eagle/checkpoint_7b/state_20 \
		--base-model-path models/vicuna-7b-v1.5 \
		--temperature 0
