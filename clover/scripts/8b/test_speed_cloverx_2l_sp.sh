export CUDA_VISIBLE_DEVICES=0
# pip install openai==0.28.0 anthropic==0.5.0
python -m clover.evaluation.gen_clo_answer_llama3_sp \
		--ea-model-path /data_train/infra/xiaobin/train/speculative-sampling/clover/clover/model_gen/llama_3_instruct_8B-h5-vloss10-layer2-lr-mlp/epoch_21 \
		--base-model-path /data_train/infra/xiaobin/train/speculative-sampling/eagle/to_copy/models/Meta-Llama-3-8B-Instruct \
		--temperature 0
