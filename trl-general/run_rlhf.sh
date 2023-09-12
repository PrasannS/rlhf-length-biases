export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29516 \
    --num_machines 1  \
    --num_processes 2 \
    train_rlhf.py --log_with=wandb \
    --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
    --dataset_name="apfarmgpt4" \
    --reward_model_name=/home/prasann/Projects/rlhf-exploration/apf/models/apfgoodtrunc \
    --adafactor=False \
    --save_freq=25 \
    --output_max_length=156 --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/apfgoodtruncppo/ \
    --init_kl_coef=0.04 --steps=1000

# # NOTE that multiple runs needs multiple device ids / whatnot
# accelerate launch --multi_gpu --config_file=default_config.yaml \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
#     --dataset_name="apfarmgpt4" \
#     --reward_model_name=/home/prasann/Projects/rlhf-exploration/apf/models/rmtruncboth \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/apfbothtruncapsftppo/ \
#     --init_kl_coef=0.04 --steps=1000

# IMPORTANT, DATASET NAME IS NOW PRETTY IMPORTANT

# accelerate launch --multi_gpu --config_file=default_config.yaml \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/home/prasann/Projects/rlhf-exploration/apf/models/sft \
#     --dataset_name="apfarmgpt4" \
#     --reward_model_name=/home/prasann/Projects/rlhf-exploration/apf/models/rmtruncboth \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/apftruncbothppo/ \
#     --init_kl_coef=0.04 --steps=1000

# accelerate launch --multi_gpu --config_file=default_config.yaml \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k \
#     --dataset_name="wgpt" \
#     --reward_model_name=/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/lenbalance \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/webgptlenbal/ \
#     --init_kl_coef=0.04 --steps=1000

# AP Farm GPT Job
# accelerate launch --multi_gpu --config_file=default_config.yaml \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k \
#     --dataset_name="apfarmgpt4" \
#     --reward_model_name=/mnt/data1/prasann/rlhf-exploration/apf/models/gptcorrect \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/apfgptppo/ \
#     --init_kl_coef=0.02 --steps=1000
