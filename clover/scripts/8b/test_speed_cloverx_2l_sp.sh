export CUDA_VISIBLE_DEVICES=0

python -m clover.evaluation.gen_clo_answer_llama3_sp \
		--ea-model-path clover/llama3-8b-h5-vloss10-layer2-lr/epoch_21/ \
		--base-model-path models/Meta-Llama-3-8B-Instruct \
		--temperature 0
