export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --num_processes 2 scripts/regressionrm_sanity.py --output_dir="checkpoints/ktorm" --dataset="data/ultrakto"