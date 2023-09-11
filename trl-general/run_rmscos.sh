# For stack sanity
# python -u rmsco_outs.py \
#     --rmname "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardsanity" \
#     --inpf "../trl-general/genouts/generated_stackmultisampset.jsonl" \
#     --device 0 \
#     --lim 800 \
#     --shuffle 10

# # For stack random only
# python -u rmsco_outs.py \
#     --rmname="/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardrandaug" \
#     --inpf="../trl-general/genouts/generated_stackmultisampset.jsonl" \
#     --device 0 \
#     --lim 800 \
#     --shuffle 10

# # # For stack DA adhoc
# python -u rmsco_outs.py \
#     --rmname="/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardda" \
#     --inpf="../trl-general/genouts/generated_stackmultisampset.jsonl" \
#     --device 0 \
#     --lim 800 \
#     --shuffle 10

# # #For webgpt
# python -u rmsco_outs.py \
#     --rmname="/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/rewardmodel" \
#     --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 1 \
#     --lim 800 \
#     --shuffle 10

# # # For stack mix
# python -u rmsco_outs.py \
#     --rmname="/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardmixed" \
#     --inpf="../trl-general/genouts/generated_stackmultisampset.jsonl" \
#     --device 1 \
#     --lim 800 \
#     --shuffle 10

# # # WebtGPT DA
# python -u rmsco_outs.py \
#     --rmname="/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/rewardrandda" \
#     --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 1 \
#     --lim 800 \
#     --shuffle 10

python -u rmsco_outs.py \
    --rmname="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/lenbalance" \
    --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
    --device 1 \
    --lim 800 \
    --shuffle 0

python -u rmsco_outs.py \
    --rmname="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmixwgpt" \
    --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
    --device 1 \
    --lim 800 \
    --shuffle 0