# Script for ultra-feedback RM training

# make sure to set to number of GPUs of your choice
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12335 scripts/train_rm.py \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
    --output_dir=checkpoints/ultrafeed/ultrarmv1 \
    --dataset="ultra" \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=1

