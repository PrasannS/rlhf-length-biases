export CUDA_VISIBLE_DEVICES=0,1,2,3


# train with preference over preference loss
torchrun --nnodes 1  --nproc_per_node 4 --master_port=12338 scripts/train_rm.py \
        --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
        --output_dir=checkpoints/ultrafeed/prefoverprefrm \
        --dataset="ultra" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=4