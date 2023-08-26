export CUDA_VISIBLE_DEVICES=0,1
torchrun --nnodes 1  --nproc_per_node 2 \
    finetune_shp.py \
    --model_path=/home/prasann/Projects/tfr-decoding/llama/llama \
    --streaming --no_gradient_checkpointing \
    --learning_rate 1e-5 --max_steps 20000 \
    --output_dir ./checkpoints/llamasft