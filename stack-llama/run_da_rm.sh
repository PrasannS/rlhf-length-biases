export CUDA_VISIBLE_DEVICES=0,1
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
torchrun --nnodes 1  --nproc_per_node 2 reward_modeling_da.py \
    --model_name=/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft