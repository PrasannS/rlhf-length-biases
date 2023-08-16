export CUDA_VISIBLE_DEVICES=0,1
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
torchrun --nnodes 1  --nproc_per_node 2 --master-port=29421 train_wgpt_rm.py \
    --model_name=/mnt/data1/prasann/prefixdecoding/tfr-decoding/llama/llama \
    --output_dir=checkpoints/wgptllama