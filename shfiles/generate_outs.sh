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

export CUDA_VISIBLE_DEVICES=4
python -u scripts/generate_outs.py \
    "models/sft10k" \
    webgpt \
    "orig" \
    "wgptextral" \
    0 500  \
    1