# export CUDA_VISIBLE_DEVICES=2
# # WebGPT Bad
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/truncppo/wgpttruncbadppo/step_150" \
#     "wgpttruncbad" \
#     0 400 \
#     1

# # WebGPT Bad
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/truncppo/wgpttruncbothppo/step_150" \
#     "wgpttruncbothearly" \
#     0 400 \
#     1

# # WebGPT Bad
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/truncppo/wgpttruncbothppo/step_300" \
#     "wgpttruncboth" \
#     0 400 \
#     1

# # WebGPT Bad
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "webgpt" \
#     "orig" \
#     "wgptonew" \
#     0 400 \
#     1

export CUDA_VISIBLE_DEVICES=2
# WebGPT Norm
python -u generate_outs.py \
    "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
    "webgpt" \
    "/home/prasann/Projects/rlhf-exploration/ppochecks/webgpt/wgptpponorm/step_125" \
    "wgptnewppo" \
    0 400 \
    1

# # APF Normal PPO new base
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "rlcd" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/rlcd/rlcdpponorm/step_150" \
#     "rlcdnewppo" \
#     0 400 \
#     1

# # APF Normal PPO new base
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "apf" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/apfgpt/apfpponorm/step_150" \
#     "apfnewppo" \
#     0 400 \
#     1

# # RLCD Trunc bad
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "rlcd" \
#     "orig" \
#     "rlcdornew" \
#     0 400 \
#     1

# # RLCD Trunc Both
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "rlcd" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/truncppo/rlcdbothppofix/step_150" \
#     "rlcdtruncboth" \
#     0 400 \
#     1

# # APF Truncate Both
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "apf" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/truncppo/apftruncbothppo/step_250" \
#     "apftruncboth" \
#     0 400 \
#     1

# # APF Truncate Both
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     "apf" \
#     "orig" \
#     "apfornew" \
#     0 400 \
#     1

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

# export CUDA_VISIBLE_DEVICES=0

# export CUDA_VISIBLE_DEVICES=1
# # generate for rlcd orig sft
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "rlcd" \
#     "orig" \
#     "rlcdsftmulti" \
#     0 800  \
#     8

# # generate for rlcd orig
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "rlcd" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/rlcd/rlcdfixed/step_100" \
#     "rlcdorig_early" \
#     0 400  \
#     1

# # generate for rlcd orig
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "rlcd" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/webgpt/fixedmixppo/step_225" \
#     "rlcdorig_lkl" \
#     0 400  \
#     1

# # TODO may need a long run for rlcd
# # generate for rlcd da
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "rlcd" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/rlcd/rlcddarmppo/step_75" \
#     "rlcddashort" \
#     0 400  \
#     1

# # generate for rlcd orig
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "apf" \
#     "orig" \
#     "apfsftmulti" \
#     0 800  \
#     8

# # generate for rlcd orig
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "apf" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/apfgpt/apfgptppo/step_100" \
#     "apforig_early" \
#     0 400  \
#     1

# # generate for rlcd orig
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "apf" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/apfgpt/apfgptppo/step_200" \
#     "apforig_conv" \
#     0 400  \
#     1

# # generate for webgpt len balanced
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/webgpt/webgptlenbal/step_200" \
#     "wgptlenbal" \
#     0 400  \
#     1

# # generate for webgpt mix 50%
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/webgpt/fixedmixppo/step_250" \
#     "wgptmix50_end" \
#     0 400  \
#     1

# # generate for webgpt mix 50%
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/ppochecks/webgpt/fixedmixppo/step_100" \
#     "wgptmix50_highkl" \
#     0 400  \
#     1

# # generate normal set for webGPT
# python -u generate_outs.py \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     "webgpt" \
#     "/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/lenconstrppo/step_200" \
#     "wgptlencons" \
#     0 800  \
#     1

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
