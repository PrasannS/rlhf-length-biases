export CUDA_VISIBLE_DEVICES=0,1
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
torchrun --nnodes 1  --nproc_per_node 2 train_shp_rm.py \
    --model_name=/home/prasann/Projects/tfr-decoding/llama/llama