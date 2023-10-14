# # DPO setup for WebGPT
# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --multi_gpu --config_file=../scripts/default_config.yaml \
#     examples/stack_llama_2/scripts/dpo_llama2.py \
#     --model_name_or_path="../models/sft10k" --output_dir="dpo" \
#     --dataset="wgpt" --log_with="wandb" 

# DPO setup for Stack
export CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file=scripts/default_config.yaml \
    dpo_exps/train_dpo.py \
    --model_name_or_path="models/stack/sft" --output_dir="dpo/dpostack" \
    --dataset="stack"

# # DPO setup for RLCD
# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=../scripts/default_config.yaml \
#     examples/stack_llama_2/scripts/dpo_llama2.py \
#     --model_name_or_path="../models/sft10k" --output_dir="dpo" \
#     --dataset="wgpt" --log_with="wandb" 