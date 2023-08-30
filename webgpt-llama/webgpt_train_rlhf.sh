export CUDA_VISIBLE_DEVICES=0,1
#export WORLD_SIZE=3
#NOTE RL TRAINED IS CONSTRAINED TO SKIP CERTAIN KINDS OF ROLLOUTS
# TODO back to normal training. use baseline
# TODO bring back multi-gpu if multi-gpu jobs are necessary
# TODO use big batches to sanity check things
accelerate launch --multi_gpu --config_file=/mnt/data1/prasann/rlhf-exploration/webgpt-llama/default_config.yaml \
    --num_machines 1  \
    --num_processes 2 \
    /home/prasann/Projects/rlhf-exploration/webgpt-llama/wgpt_train_rl.py --log_with=wandb \
    --model_name=/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k/ \
    --reward_model_name=/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/mixrm/ \
    --adafactor=False \
    --tokenizer_name=/mnt/data1/prasann/rlhf-exploration/webgpt-llama/models/sft10k \
    --save_freq=25 \
    --output_max_length=156 --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/mixrmppo/ \
    --init_kl_coef=0.005 --steps=1000

#v3, aimed at greater stability (4 gpu run, 2 ppo epochs per thing)
#lowp is 0.8 top p, should constrain exploration a bit