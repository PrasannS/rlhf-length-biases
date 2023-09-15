# export CUDA_VISIBLE_DEVICES=0,1

# #export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29516 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
#     --dataset_name="rlcd" \
#     --reward_model_name=/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcdmidtrunc \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rlcdmidtruncv3/ \
#     --init_kl_coef=0.04 --steps=1000
    #--reward_baseline=1.5

# export CUDA_VISIBLE_DEVICES=2,3

# #export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
#     --dataset_name="apfarmgpt4" \
#     --reward_model_name=/home/prasann/Projects/rlhf-exploration/apf/models/apfrandcarto \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/apfrandcartoppo/ \
#     --init_kl_coef=0.04 --steps=151

export CUDA_VISIBLE_DEVICES=0,1

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="rlcd" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/rlcdbothcut2 \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rlcdbothppov2/ \
#     --init_kl_coef=0.04 --steps=151

# export CUDA_VISIBLE_DEVICES=2,3

# #export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29518 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="rlcd" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/rlcdleftonly2 \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rlcdleftonlyppov2/ \
#     --init_kl_coef=0.04 --steps=151

# export CUDA_VISIBLE_DEVICES=4,5
# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29519 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="wgpt" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/wgptgoodcut2 \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/wgptgoodppov2/ \
#     --init_kl_coef=0.04 --steps=151

# export CUDA_VISIBLE_DEVICES=4,5

# #export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29519 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="wgpt" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/wgptbothcut2 \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/wgptbothppo2/ \
#     --init_kl_coef=0.04 --steps=151

export CUDA_VISIBLE_DEVICES=6,7
accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29520 \
    --num_machines 1  \
    --num_processes 2 \
    train_rlhf.py --log_with=wandb \
    --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
    --dataset_name="wgpt" \
    --reward_model_name=/home/prasann/Projects/rlhf-exploration/apf/models/wgptrandcarto \
    --adafactor=False \
    --save_freq=25 \
    --output_max_length=156 --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/wgptrandcartoppo/ \
    --init_kl_coef=0.04 --steps=151

# export CUDA_VISIBLE_DEVICES=2,3

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29502 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="rlcd" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/rmtruncrightrlcd \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rlcdrightonlyppo/ \
#     --init_kl_coef=0.04 --steps=1000
#     #--reward_baseline=1.5

# export CUDA_VISIBLE_DEVICES=0,1

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29501 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="rlcd" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/rmtruncleftrlcd \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rlcdleftonlyppo/ \
#     --init_kl_coef=0.04 --steps=1000
#     #--reward_baseline=1.5

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29516 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="rlcd" \
#     --reward_model_name=/mnt/data1/prasann/rlhf-exploration/rlcd-llama/models/rlcdgoodtrunc \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rlcdgoodtruncvv3/ \
#     --init_kl_coef=0.04 --steps=1000
#     #--reward_baseline=1.5

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
