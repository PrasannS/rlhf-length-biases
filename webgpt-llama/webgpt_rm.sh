export CUDA_VISIBLE_DEVICES=2,3
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
torchrun --nnodes 1  --nproc_per_node 2 --master-port=29421 wgpt_modeling_da.py \
    --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
    --output_dir=checkpoints/webgptrandda