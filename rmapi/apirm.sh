# NOTE that this endpoint needs to match with the desired reward model 
# TODO maybe specify an extra port somehow

# export CUDA_VISIBLE_DEVICES=0

# python -u rmapi/rmapi.py \
#     --model_name=facebook/opt-125m \
#     --dataset_name="/u/prasanns/research/rlhf-length-biases/data/bowsynth50knozeros" \
#     --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/expbow50 \
#     --save_freq=25 \
#     --max_length=256 --batch_size=16 \
#     --seed=0 --learning_rate=5e-5 \
#     --output_dir=checkpoints/trainablegapheur \
#     --trainable --goldreward="function:bagofwords" \
#     --logfile="outputs/dynarmlogs/trainheuractive.jsonl" \
#     --trainheur \
#     --port=5001

# python -u rmapi/rmapi.py \
#     --dataset_name="ultra" \
#     --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/ultra50krm \
#     --save_freq=25 \
#     --max_length=256 --batch_size=16 \
#     --learning_rate=1e-4 \
#     --output_dir=""

export CUDA_VISIBLE_DEVICES=0
python -u rmapi/rmapi.py \
    --dataset_name="ultra" \
    --reward_model_name=/u/prasanns/research/rlhf-length-biases/models/rewards/13btruncrm \
    --save_freq=25 \
    --max_length=256 --batch_size=16 \
    --learning_rate=1e-4 \
    --output_dir="" \
    --port=5000