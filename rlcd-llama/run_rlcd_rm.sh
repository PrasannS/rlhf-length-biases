export CUDA_VISIBLE_DEVICES=0,1
# NOTE IMPORTANT, CHECK FILE NAME BEFORE RUNNING
torchrun --nnodes 1  --nproc_per_node 2 --master_port=12345 train_rlcd_rm.py \
    --model_name=/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k