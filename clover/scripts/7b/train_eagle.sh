
export WANDB_API_KEY="" # set wandb key
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch -m --mixed_precision=bf16 --main_process_port 29501 clover.train.main \
    --basepath models/vicuna-7b-v1.5 \
    --evaldata datasets/Spe_eval_merge_7b/sharegpt_0_479_mubf16 \
    --tmpdir datasets/ShareGPT_Vicuna_onnorm/sharegpt_0_67999_mubf16 \
    --cpdir eagle/checkpoint_7b \
    --configpath train/vicuna_7B_config.json \
    --gradient_checkpointing True \
    --bs 4 
