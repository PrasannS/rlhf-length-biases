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
    "models/stack/sft" \
    stack \
    "/u/prasanns/research/rlhf-length-biases/checkpoints/stackapibigrmstep_225" \
    "stack13bppo" \
    0 500  \
    1