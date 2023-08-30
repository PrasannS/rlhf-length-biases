export CUDA_VISIBLE_DEVICES=0,1

python -u make_adv_dset.py \
    "../trl-general/generated_wgptsft.jsonl" \
    "attackouts/wgpt/wgptorig.jsonl" \
    "wgpt" \
    "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmodel"

python -u make_adv_dset.py \
    "../trl-general/generated_wgptsft.jsonl" \
    "attackouts/wgpt/wgptrandda.jsonl" \
    "wgpt" \
    "/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardrandda"