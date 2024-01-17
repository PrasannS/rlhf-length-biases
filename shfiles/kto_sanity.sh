export CUDA_VISIBLE_DEVICES=2,3
accelerate launch --multi_gpu --num_processes 2 scripts/regressionrm_sanity.py \
    --output_dir="checkpoints/expbowktosanity" --dataset="data/expbowregression"