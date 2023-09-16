# # Try again full pipeline for new augmentation 
export CUDA_VISIBLE_DEVICES=0,1
# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12335 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcddiagrmfix/ \
#     --dataset="rlcd" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/diagboth.json"

# python merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-exploration/trl-general/checkpoints/rlcddiagrmfix/checkpoint-9500" \
#     --base_model_name="/u/prasanns/research/rlhf-exploration/models/sft" \
#     --output_name="/u/prasanns/research/rlhf-exploration/models/rewards/rlcddiagfix"

accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29521 \
    --num_machines 1  \
    --num_processes 2 \
    train_rlhf.py --log_with=wandb \
    --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
    --dataset_name="rlcd" \
    --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/rlcddiagfix \
    --adafactor=False \
    --save_freq=25 \
    --output_max_length=156 --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/rlcddiagfixppo/ \
    --init_kl_coef=0.04 --steps=250

# export CUDA_VISIBLE_DEVICES=2,3
# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12336 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/rlcdmixnew/ \
#     --dataset="rlcd" \
#     --mix_ratio=0.50 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=2

# python merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-exploration/trl-general/checkpoints/rlcdmixnew/checkpoint-10000" \
#     --base_model_name="/u/prasanns/research/rlhf-exploration/models/sft" \
#     --output_name="/u/prasanns/research/rlhf-exploration/models/rewards/rlcdmixnew"

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29522 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="rlcd" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/rlcdmixnew \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/rlcdmixnewppo/ \
#     --init_kl_coef=0.04 --steps=151

export CUDA_VISIBLE_DEVICES=6,7
# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12337 train_rm.py \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft \
#     --output_dir=./checkpoints/apfmixnew/ \
#     --dataset="apfarmgpt4" \
#     --mix_ratio=0.50 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=2

# python merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-exploration/trl-general/checkpoints/apfmixnew/checkpoint-6500" \
#     --base_model_name="/u/prasanns/research/rlhf-exploration/models/sft" \
#     --output_name="/u/prasanns/research/rlhf-exploration/models/rewards/apfmixnew"

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29525 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/u/prasanns/research/rlhf-exploration/models/sft10k \
#     --dataset_name="apfarmgpt4" \
#     --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/apfmixnew \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/apfmixnewppo/ \
#     --init_kl_coef=0.04 --steps=251