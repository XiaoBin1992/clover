
export WANDB_API_KEY="" # set wandb key
export CUDA_VISIBLE_DEVICES=0,1,4,5
#bash clover/scripts/8b/train_clover2.sh  > info2.log 2>&1 &

accelerate launch -m --mixed_precision=bf16 --main_process_port 29613 clover.train.main_clover2 \
    --basepath /data_train/infra/xiaobin/train/speculative-sampling/eagle/to_copy/models/Meta-Llama-3-8B-Instruct \
    --evaldata /data_train/infra/xiaobin/train/speculative-sampling/eagle/to_copy/datasets/Spe_eval_merge_llama3_8b/sharegpt_0_479_mubf16 \
    --tmpdir /data_train/infra/xiaobin/train/speculative-sampling/eagle/dataset/sharegpt_0_67999_mubf16 \
    --cpdir clover/model_gen/llama_3_instruct_8B-h5-vloss10-layer2-lr \
    --configpath clover/train/llama3_instruct_8B_config.json \
    --gradient_checkpointing True \
    --bs 4
