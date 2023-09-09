export CUDA_VISIBLE_DEVICES=2,3

# dataset can be [wgpt, rlcd, stack, apfarm]


# webgpt run with new base model and data carto
torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
    --model_name=/home/prasann/Projects/rlhf-exploration/apf/models/sft \
    --output_dir=./checkpoints/webgptrda_carto/ \
    --dataset="wgpt" \
    --mix_ratio=0 \
    --rand_ratio=0.2 \
    --balance_len=0 \
    --num_train_epochs=10

# # rlcd run with len balancing
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/rlcdlenbal/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=1 \
#     --num_train_epochs=2

# # rlcd run with lower mix ratio
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/rlcdmix20/ \
#     --dataset="rlcd" \
#     --mix_ratio=0.2 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=2

# # webgpt run with lower mix ratio
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/webgptmix20/ \
#     --dataset="wgpt" \
#     --mix_ratio=0.2 \
#     --rand_ratio=0 \
#     --balance_len=0

# # stack run with balancingy
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/stacklenbal/ \
#     --dataset="stack" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=1 \
#     --num_train_epochs=2

# # stack run with lower mix ratio
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12346 train_rm.py \
#     --model_name=/home/prasann/Projects/tfr-decoding/llama/llama \
#     --output_dir=./checkpoints/stackmix20/ \
#     --dataset="stack" \
#     --mix_ratio=0.2 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=2

