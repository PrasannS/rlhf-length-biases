# export CUDA_VISIBLE_DEVICES=2

# # # For WebGPT new model
# # python -u generate_outs.py \
# #     "/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
# #     "webgpt" \
# #     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/wgptapsft/step_125" \
# #     "wgptppoorig" \
# #     0 200

# # # For WebGPT DA model
# python -u generate_outs.py \
#     "/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/wgptrdappo/step_150" \
#     "wgptpporda" \
#     0 200

export CUDA_VISIBLE_DEVICES=2,3

# generate normal set for webGPT
python -u generate_outs.py \
    "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
    "webgpt" \
    "/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/lenconstrppo/step_200" \
    "wgptlencons" \
    0 800  \
    1

# To generate from the bigdata stack RM
# python -u generate_outs.py \
#     "/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/sft" \
#     "stack" \
#     "/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/bigrmppo/step_150" \
#     "stackbig150" \
#     0 200

# # To regen stack DA 125 (remember it has a different, old SFT model)
# python -u generate_outs.py \
#     "/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/sft" \
#     "stack" \
#     "/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/usemodels/da/step_125" \
#     "stackda125" \
#     0 200

# # Regen original SFT 
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     "stack" \
#     "orig" \
#     "stacksft" \
#     0 200

# To regen stack sanity 

# To regen stack SFT


# For original apfarm model on WebGPT
# python -u generate_outs.py \
#     "/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#     "webgpt" \
#     "orig" \
#     "wgptsft" \
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
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "orig" \
#     "wgptmultisampset" \
#     0 800  \
#     8 

# multi-sample set generation for stack
# python -u generate_outs.py \
#     "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/sft" \
#     "stack" \
#     "orig" \
#     "stackmultisampset" \
#     0 800 \
#     8
