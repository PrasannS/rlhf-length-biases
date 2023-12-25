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

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     ultra \
#     "/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_200" \
#     "ultr200" \
#     0 200  \
#     4

# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     ultra \
#     "/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_400" \
#     "ultra400" \
#     0 200  \
#     4

# export CUDA_VISIBLE_DEVICES=3
# python -u scripts/generate_outs.py \
#     "models/sft10k" \
#     ultra \
#     "/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_100" \
#     "longppo_big100" \
#     6400 8000  \
#     4


export CUDA_VISIBLE_DEVICES=3
python -u scripts/generate_outs.py \
    "models/sft10k" \
    ultra \
    "/home/prasann/Projects/rlhf-length-biases/checkpoints/ultrappolongrun/step_125" \
    "longppo_big125v2" \
    8400 9600  \
    4
