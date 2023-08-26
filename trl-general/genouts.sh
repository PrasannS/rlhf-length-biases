export CUDA_VISIBLE_DEVICES=1

# For WebGPT new model
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/checkpoints/wgptapsft/step_125" \
#     "wgptppoorig" \
#     0 800

# For original apfarm model on WebGPT
python -u generate_outs.py \
    "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
    "webgpt" \
    "orig" \
    "wgptsft" \
    0 800