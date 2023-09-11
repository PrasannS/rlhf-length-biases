export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes 1  --nproc_per_node 2 train_sftmod.py --model_path=/home/prasann/Projects/tfr-decoding/llama/llama --streaming --no_gradient_checkpointing --learning_rate 1e-5 --max_steps 5000 --output_dir ./checkpoints/alpsftllama
