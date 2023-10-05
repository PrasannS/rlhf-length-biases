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