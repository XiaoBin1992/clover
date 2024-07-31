
export WANDB_API_KEY="" # set wandb key
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch -m --mixed_precision=bf16 --main_process_port 29501 clover.train.main \
    --basepath models/Meta-Llama-3-8B-Instruct \
    --evaldata datasets/Spe_eval_merge_llama3_8b/sharegpt_0_479_mubf16 \
    --tmpdir datasets/ShareGPT_llama3ins8b_onnorm_bf16/sharegpt_0_67999_mubf16 \
    --cpdir eagle/checkpoint_8b \
    --configpath train/llama3_instruct_8B_config.json \
    --gradient_checkpointing True \
    --bs 4 
