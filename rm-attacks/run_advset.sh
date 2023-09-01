export CUDA_VISIBLE_DEVICES=0,1

python -u make_adv_dset.py \
    "wgptbase.jsonl" \
    "attackouts/wgptorig.jsonl" \
    "stack" \
    "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/rewardmodel"

python -u make_adv_dset.py \
    "wgptbase.jsonl" \
    "attackouts/wgptda.jsonl" \
    "stack" \
    "/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/rewardrandda"
    
python -u make_adv_dset.py \
    "stackbase.jsonl" \
    "attackouts/stacksanity.jsonl" \
    "stack" \
    "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardsanity"

python -u make_adv_dset.py \
    "stackbase.jsonl" \
    "attackouts/stackda.jsonl" \
    "stack" \
    "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardda"

python -u make_adv_dset.py \
    "stackbase.jsonl" \
    "attackouts/stackrandaug.jsonl" \
    "stack" \
    "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardrandaug"

python -u make_adv_dset.py \
    "stackbase.jsonl" \
    "attackouts/stackmix.jsonl" \
    "stack" \
    "/mnt/data1/prasann/rlhf-exploration/stack-llama/models/rewardmixed"

