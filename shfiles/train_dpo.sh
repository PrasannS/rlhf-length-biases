# # DPO setup for WebGPT
# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/sft10k" --output_dir="dpo/dpowgpt" \
#     --dataset="wgpt"

# DPO setup for Stack
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file=scripts/default_single.yaml \
#     dpo_exps/train_dpo.py \
#     --model_name_or_path="models/stack/sft" --output_dir="dpo/dpostack" \
#     --dataset="stack"

# # DPO setup for RLCD
export CUDA_VISIBLE_DEVICES=1
accelerate launch --config_file=scripts/default_single.yaml \
    dpo_exps/train_dpo.py \
    --model_name_or_path="models/sft10k" --output_dir="dpo/dporlcd" \
    --dataset="rlcd"
