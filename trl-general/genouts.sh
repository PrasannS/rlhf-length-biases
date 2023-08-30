export CUDA_VISIBLE_DEVICES=1

# For WebGPT new model
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/checkpoints/wgptapsft/step_125" \
#     "wgptppoorig" \
#     0 800

# For original apfarm model on WebGPT
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "orig" \
#     "wgptsft" \
#     0 800


# For DA model WebGPT
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/checkpoints/wgptrdappo/step_150" \
#     "webgptdappo" \
#     0 200

# For big stack model
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/sft" \
#     "stack" \
#     "/mnt/data1/prasann/rlhf-exploration/stack-llama/checkpoints/bigrmppo/step_150" \
#     "stackbig150" \
#     0 200

# For DA webgptlkl125
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/checkpoints/wgptrdappov2/step_125" \
#     "webgptdappov2125" \
#     0 200

# For DA webgptlkl250
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/checkpoints/wgptrdappov2/step_250" \
#     "webgptdappov250" \
#     0 400  \
#     8

# multi-sample set generation for webgpt
python -u generate_outs.py \
    "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
    "webgpt" \
    "orig" \
    "wgptmultisampset" \
    0 800  \
    8 

# multi-sample set generation for stack
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/sft" \
#     "stack" \
#     "orig" \
#     "stackmultisampset" \
#     0 800 \
#     8