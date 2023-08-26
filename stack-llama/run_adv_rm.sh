export CUDA_VISIBLE_DEVICES=2,3
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
torchrun --nnodes 1  --nproc_per_node 2 --master_port=12345 reward_adv_da.py \
    --model_name=/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft \
    --output_dir=./checkpoints/advmse/