export CUDA_VISIBLE_DEVICES=0

python -m clover.evaluation.gen_clo_answer_vicuna_sp \
		--ea-model-path /clover/vicuna-7b-v1.5-h5-vloss10-layer2-lr/epoch_21/ \
		--base-model-path models/vicuna-7b-v1.5 \
		--temperature 0
