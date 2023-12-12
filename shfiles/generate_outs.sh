# export CUDA_VISIBLE_DEVICES=0
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     webgpt \
#     "dpo/dpowgpt/checkpoint-4000" \
#     "wgptdpo" \
#     0 500  \
#     1

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     rlcd \
#     "dpo/dporlcd/checkpoint-4000" \
#     "rlcddpo" \
#     0 500  \
#     1

export CUDA_VISIBLE_DEVICES=5
python -u scripts/generate_outs.py \
    "facebook/opt-125m" \
    ultra \
    "/u/prasanns/research/rlhf-length-biases/dpo/dpobow/checkpoint-3000" \
    "dpobow3k" \
    0 100  \
    4

# python -u scripts/generate_outs.py \
#     "facebook/opt-125m" \
#     ultra \
#     "/u/prasanns/research/rlhf-length-biases/dpo/dponoun/checkpoint-42000" \
#     "dponoun" \
#     0 100  \
#     4
