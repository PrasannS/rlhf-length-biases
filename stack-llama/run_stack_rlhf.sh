export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
#HACK running constrained version, pay attention

accelerate launch --multi_gpu --config_file=/mnt/data1/prasann/prefixdecoding/tfr-decoding/trlx_train/trl-stack/default_config.yaml \
    --num_machines 1  \
    --num_processes 2 \
    rl_training.py --log_with=wandb \
    --model_name=/mnt/data1/prasann/prefixdecoding/tfr-decoding/trlx_train/trl-stack/models/sft \
    --reward_model_name=/mnt/data1/prasann/prefixdecoding/tfr-decoding/trlx_train/trl-stack/models/rewardsanity \
    --adafactor=False \
    --tokenizer_name=/mnt/data1/prasann/prefixdecoding/tfr-decoding/trlx_train/trl-stack/models/sft \
    --save_freq=25 \
    --output_max_length=156 --batch_size=32 \
    --gradient_accumulation_steps=1 --batched_gen=True \
    --ppo_epochs=1 --seed=200 --learning_rate=1.4e-5 \
    --early_stopping=True --output_dir=checkpoints/sanity1epoch/ \
    --init_kl_coef=0.2

# TODO rename baseline2 to allconstr