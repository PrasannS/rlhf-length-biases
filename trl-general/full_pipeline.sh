# # Try again full pipeline for new augmentation 
# export CUDA_VISIBLE_DEVICES=0,1
# # dataset can be [wgpt, rlcd, stack, apfarm]
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12335 train_rm.py \
#     --model_name=/home/prasann/Projects/rlhf-exploration/apf/models/sft \
#     --output_dir=./checkpoints/apfmidcut/ \
#     --dataset="apfarmgpt4" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/apfmidcut.json"

# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/apfmidcut/checkpoint-4000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/apfmidcut"

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29521 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
#     --dataset_name="apfarmgpt4" \
#     --reward_model_name=/home/prasann/Projects/rlhf-exploration/apf/models/apfmidcut \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/apfmidcutppo/ \
#     --init_kl_coef=0.04 --steps=300

# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nnodes 1  --nproc_per_node 2 --master_port=12336 train_rm.py \
#     --model_name=/home/prasann/Projects/rlhf-exploration/apf/models/sft \
#     --output_dir=./checkpoints/wgptmidcut/ \
#     --dataset="wgpt" \
#     --mix_ratio=0 \
#     --rand_ratio=0 \
#     --balance_len=0 \
#     --num_train_epochs=4 \
#     --carto_file="truncvals/wgptmidcut.json"

# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/trl-general/checkpoints/wgptmidcut/checkpoint-4000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/apf/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/wgptmidcut"

# accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29520 \
#     --num_machines 1  \
#     --num_processes 2 \
#     train_rlhf.py --log_with=wandb \
#     --model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k \
#     --dataset_name="wgpt" \
#     --reward_model_name=/home/prasann/Projects/rlhf-exploration/apf/models/wgptmidcut \
#     --adafactor=False \
#     --save_freq=25 \
#     --output_max_length=156 --batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
#     --early_stopping=False --output_dir=checkpoints/wgptmidcutppo/ \
#     --init_kl_coef=0.04 --steps=300

export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes 1  --nproc_per_node 2 --master_port=29518 train_rm.py \
    --model_name=/u/prasanns/research/rlhf-exploration/models/stack/sft \
    --output_dir=./checkpoints/stackbothcut/ \
    --dataset="stack" \
    --mix_ratio=0 \
    --rand_ratio=0 \
    --balance_len=0 \
    --num_train_epochs=2 \
    --carto_file="truncvals/stackboth.json"

python merge_peft_adapter.py \
    --adapter_model_name="/u/prasanns/research/rlhf-exploration/trl-general/checkpoints/stackbothcut/checkpoint-8000" \
    --base_model_name="/u/prasanns/research/rlhf-exploration/models/stack/sft" \
    --output_name="/u/prasanns/research/rlhf-exploration/models/rewards/stackbothcut"

export CUDA_VISIBLE_DEVICES=2,3
accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29518 \
    --num_machines 1  \
    --num_processes 2 \
    train_rlhf.py --log_with=wandb \
    --model_name=/u/prasanns/research/rlhf-exploration/models/stack/sft \
    --dataset_name="stack" \
    --reward_model_name=/u/prasanns/research/rlhf-exploration/models/rewards/stackbothcut \
    --adafactor=False \
    --save_freq=25 \
    --output_max_length=216 --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/stackbothcutppo/ \
    --init_kl_coef=0.04 --steps=1000 