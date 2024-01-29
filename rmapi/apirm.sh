# NOTE that this endpoint needs to match with the desired reward model 
# TODO maybe specify an extra port somehow

python -u rmapi/rmapi.py --log_with=wandb \
    --model_name=facebook/opt-125m \
    --dataset_name="/u/prasanns/research/rlhf-length-biases/data/bowsynth50knozeros" \
    --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50 \
    --adafactor=False \
    --save_freq=25 \
    --max_length=256 --batch_size=16 \
    --gradient_accumulation_steps=1 \
    --ppo_epochs=1 --seed=0 --learning_rate=1e-4 \
    --early_stopping=False --output_dir=checkpoints/trainablebowv1 \
    --init_kl_coef=0.04 --steps=1000 \
    --

export CUDA_VISIBLE_DEVICES=0

# python -u rmapi/rmapi.py \
#     --dataset_name="ultra" \
#     --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/ultra50krm \
#     --save_freq=25 \
#     --max_length=256 --batch_size=16 \
#     --learning_rate=1e-4 \
#     --output_dir=""

# export CUDA_VISIBLE_DEVICES=1
# python -u rmapi/rmapi.py \
#     --dataset_name="ultra" \
#     --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/ultrasmalldistrm \
#     --save_freq=25 \
#     --max_length=256 --batch_size=16 \
#     --learning_rate=1e-4 \
#     --output_dir="" \
#     --port=5001