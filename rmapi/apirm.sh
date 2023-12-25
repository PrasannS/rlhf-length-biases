export CUDA_VISIBLE_DEVICES=0
# NOTE that this endpoint needs to match with the desired reward model 
# TODO maybe specify an extra port somehow
python rmapi/rmapi.py --log_with=wandb \
    --model_name=meta-llama/Llama-2-13b-hf \
    --dataset_name="ultra" \
    --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/ultra13b \
    --adafactor=False \
    --save_freq=25 \
    --output_max_length=256 --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1.4e-5 \
    --early_stopping=False --output_dir=checkpoints/wgptreward \
    --init_kl_coef=0.04 --steps=1000