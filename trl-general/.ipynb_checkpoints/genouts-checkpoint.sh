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

export CUDA_VISIBLE_DEVICES=3

# To generate from the bigdata stack RM
python -u generate_outs.py \
    "/home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/sft" \
    "stack" \
    "/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/bigrmppo/step_150" \
    "stackbig150" \
    0 200

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