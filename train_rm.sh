export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12335 scripts/train_rm.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/llama \
#         --output_dir=checkpoints/harmless/harmlessrm \
#         --dataset="harmless" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=5

# Do cross eval
# torchrun --nnodes 1  --nproc_per_node 4 --master_port=12337 scripts/stack_cross_eval.py \
#         --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
#         --output_dir=checkpoints/stackrms/stack_workplace \
#         --dataset="stack_workplace" \
#         --rand_ratio=0 \
#         --balance_len=0 \
#         --num_train_epochs=1

torchrun --nnodes 1  --nproc_per_node 6 --master_port=12337 scripts/train_rm.py \
        --model_name=meta-llama/Llama-2-13b-hf \
        --output_dir=checkpoints/ultrafeed/bigmag \
        --dataset="ultra" \
        --rand_ratio=0 \
        --balance_len=0 \
        --num_train_epochs=2