# export CUDA_VISIBLE_DEVICES=0,1

#dataset can be [wgpt, rlcd, stack, apfarm]

# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12348 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcd_carto/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0.0 \
#     --balance_len=0 \
#     --num_train_epochs=5
# export CUDA_VISIBLE_DEVICES=0,1
# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12333 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/webgpttruncboth/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/webgptboth.json"

# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12335 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdtruncbad/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/webgptbad.json"

# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12349 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdtruncboth/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=3 \
#     --carto_file="truncvals/rlcdboth.json"

export CUDA_VISIBLE_DEVICES=6,7
torchrun --nnodes 1  --nproc_per_node 2 --master_port=12350 train_rm.py \
    --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
    --output_dir=./checkpoints/rlcdtruncbad/ \
    --dataset="rlcd" \
    --mix_ratio=0 \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=3 \
    --carto_file="truncvals/rlcdbad.json"


# # # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12347 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/stack/sft \
#     --output_dir=./checkpoints/stack_carto/ \
#     --dataset="stack" \
#     --mix_ratio=0 \
#     --rand_ratio=0.0 \
#     --balance_len=0 \
#     --num_train_epochs=5

# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12347 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/stack/sft \
#     --output_dir=./checkpoints/stackda_carto/ \
#     --dataset="stack" \
#     --mix_ratio=0 \
#     --rand_ratio=0.2 \
#     --balance_len=0 \
#     --num_train_epochs=5

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

