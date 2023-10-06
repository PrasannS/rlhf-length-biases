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

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/lenbalance" \
#     --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 1 \
#     --lim 800 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmixwgpt" \
#     --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 1 \
#     --lim 800 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/apfgoodcut2" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_apfsftmulti.jsonl" \
#     --device 2 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/apfnormal" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_apfsftmulti.jsonl" \
#     --device 2 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/wgptgoodcut2" \
#     --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 2 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/wgptbothcut2" \
#     --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 2 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/wgptnormal" \
#     --inpf="../trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 2 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/rlcdbothcut2" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_rlcdsftmulti.jsonl" \
#     --device 3 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/rlcdgoodcut2" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_rlcdsftmulti.jsonl" \
#     --device 3 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/rlcdleftonly2" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_rlcdsftmulti.jsonl" \
#     --device 3 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/rlcdmidcut2" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_rlcdsftmulti.jsonl" \
#     --device 3 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/models/rewards/rlcdnormal" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_rlcdsftmulti.jsonl" \
#     --device 3 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/apf/models/apfrandcarto" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_apfsftmulti.jsonl" \
#     --device 0 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/apf/models/wgptrandcarto" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_wgptmultisampset.jsonl" \
#     --device 1 \
#     --lim 600 \
#     --shuffle 0
# python -u rmsco_outs.py \
#     --rmname="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcdlenbal" \
#     --inpf="/home/prasann/Projects/rlhf-exploration/trl-general/genouts/generated_rlcdsftmulti.jsonl" \
#     --device 1 \
#     --lim 600 \
#     --shuffle 0

# python -u rmsco_outs.py \
#     --rmname="/u/prasanns/research/rlhf-exploration/models/rewards/stackbothcut" \
#     --inpf="/u/prasanns/research/rlhf-exploration/trl-general/genouts/generated_stackmultisampset.jsonl" \
#     --device 2 \
#     --lim 600 \
#     --shuffle 0

python -u rmsco_outs.py \
    --rmname="/u/prasanns/research/rlhf-exploration/models/rewards/wgptbalancecorrect" \
    --inpf="/u/prasanns/research/rlhf-exploration/trl-general/genouts/generated_wgptmultisampset.jsonl" \
    --device 3 \
    --lim 600 \
    --shuffle 0
