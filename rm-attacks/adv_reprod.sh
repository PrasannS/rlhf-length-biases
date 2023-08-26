export CUDA_VISIBLE_DEVICES=0,1
python -u repeat_adv_dset.py "../stack-llama/models/rewardadvda/" \
    "../outputs/augdata/augstackv1.jsonl" \
    "attackouts/reprodcheck/advreprodboth.jsonl"