

torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/ultra_cross_eval.py \
        --model_name=meta-llama/Llama-2-13b-hf \
        --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/ultrafeed/bigmag/_peft_last_checkpoint \
        --dataset="ultra" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=1