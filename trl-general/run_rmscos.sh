# For stack sanity
# python -u rmsco_outs.py \
#     /home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardsanity \
#     "../trl-general/genouts/generated_stackmultisampset.jsonl" \
#     0

# For stack random only
python -u rmsco_outs.py \
    /home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardrandaug \
    "../trl-general/genouts/generated_stackmultisampset.jsonl" \
    0

# # For stack DA adhoc
# python -u rmsco_outs.py \
#     /home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/rewardda \
#     "../trl-general/genouts/generated_stackmultisampset.jsonl" \
#     2

# #For webgpt
# python -u rmsco_outs.py \
#     /home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmodel \
#     "../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     2

# # For stack mix
# python -u rmsco_outs.py \
#     /home/prasann/Projects/tfr-decoding/trlx_train/trl-stack/models/rewardmixed \
#     "../trl-general/genouts/generated_stackmultisampset.jsonl" \
#     3

# # WebtGPT DA
# python -u rmsco_outs.py \
#     /home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardrandda \
#     "../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     3

