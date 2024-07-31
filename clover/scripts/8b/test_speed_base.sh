export CUDA_VISIBLE_DEVICES=0

python -m clover.evaluation.gen_baseline_answer_llama3 \
		--ea-model-path eagle/checkpoint_8b/state_20 \
		--base-model-path models/Meta-Llama-3-8B-Instruct \
		--temperature 0
