
export WANDB_API_KEY="" # set wandb key
#export CUDA_VISIBLE_DEVICES=0,1,2,3
# ps -ef|grep -E "eagle|tail|wandb"|awk -F " " '{print $2}'| xargs kill -9
#bash clover/scripts/7b/train_clover2.sh  > info.log 2>&1 &

accelerate launch -m --mixed_precision=bf16 --main_process_port 29613 clover.train.main_clover2 \
    --basepath /data_train/infra/glj/models/vicuna-7b-v1.5 \
    --evaldata /data_train/infra/glj/datasets/Spe_eval_merge_7b/sharegpt_0_479_mubf16 \
    --tmpdir /data_train/infra/glj/datasets/SharedGPT_7b/sharegpt_0_67999_mubf16 \
    --cpdir clover/model_gen/7B-h5-vloss10-layer2-lr-mlp \
    --configpath clover/train/vicuna_7B_config.json \
    --gradient_checkpointing True \
    --bs 2