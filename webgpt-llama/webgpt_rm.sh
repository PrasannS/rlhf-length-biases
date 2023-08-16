export CUDA_VISIBLE_DEVICES=2,3
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
torchrun --nnodes 1  --nproc_per_node 2 --master-port=29421 train_wgpt_rm.py \
    --model_name=/home/prasann/Projects/tfr-decoding/llama/llama