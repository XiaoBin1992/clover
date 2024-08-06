export CUDA_VISIBLE_DEVICES=0

python -m clover.evaluation.gen_ea_answer_llama3_sp \
		--ea-model-path /data_train/infra/glj/models/eagle/clover1_llama3ins-8b/state_20 \
		--base-model-path /data_train/infra/xiaobin/train/speculative-sampling/eagle/to_copy/models/Meta-Llama-3-8B-Instruct \
		--temperature 0
