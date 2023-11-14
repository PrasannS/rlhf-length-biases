

# torchrun --nnodes 1  --nproc_per_node 8 --master_port=12337 scripts/ultra_cross_eval.py \
#         --model_name=meta-llama/Llama-2-13b-hf \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/ultrafeed/bigmag/_peft_last_checkpoint \
#         --dataset="ultra" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1

# export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12339 scripts/ultra_cross_eval.py \
#         --model_name=facebook/opt-125m \
#         --output_dir=/u/prasanns/research/rlhf-length-biases/checkpoints/synth/noun90nomag/_peft_last_checkpoint \
#         --dataset="/u/prasanns/research/rlhf-length-biases/data/synthnoun90" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1



        