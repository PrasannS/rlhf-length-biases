#export CUDA_VISIBLE_DEVICES=0,1

# accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 2 \
#     scripts/train_rlhf.py --log_with=wandb \
#     --model_name=/home/prasann/Projects/rlhf-length-biases/models/sft10k \
#     --dataset_name="ultra" \
#     --reward_model_name=/home/prasann/Projects/rlhf-length-biases/models/rewards/ultrarm \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=256 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/ultra/ultrappo \
#     --init_kl_coef=0.04 --steps=1000 

# in case something happens while I'm sleeping w GPU mem, bring things down and try again
# NOTE I pushed the LR down a bit
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
accelerate launch --multi_gpu --config_file=scripts/default_config.yaml --main_process_port=29518 \
    --num_machines 1  \
    --num_processes 7 \
    scripts/train_rlhf.py --log_with=wandb \
    --model_name=/u/prasanns/research/rlhf-length-biases/models/stack/sft \
    --dataset_name="stack" \
    --reward_model_name="http://127.0.0.1:5000/predict" \
    --adafactor=False \
    --save_freq=25 \
    --output_max_length=256 --batch_size=16 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/stackapibigrm \
    --init_kl_coef=0.02 --steps=1000 
