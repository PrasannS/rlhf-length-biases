export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes 1  --nproc_per_node 4 train_sftmod.py --model_path=EleutherAI/gpt-neo-1.3B --streaming --no_gradient_checkpointing --learning_rate 1e-5 --max_steps 5000 --output_dir ./checkpoints/alpsft
