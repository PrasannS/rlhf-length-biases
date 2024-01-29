# export CUDA_VISIBLE_DEVICES=1
# python -u scripts/tunesft.py \
#     --dataset_name="data/einstein/bigsfteinstein" \
#     --output_dir="checkpoints/einsteinoptcorrectedbig" \
#     --save_steps=500 \
#     --num_train_epochs=2 \
#     --learning_rate=5e-4 \
#     --model_name=facebook/opt-125m \
#     --per_device_train_batch_size=16 \
#     --per_device_eval_batch_size=16 \
#     --warmup_steps=100 \
#     --logging_steps=10

export CUDA_VISIBLE_DEVICES=2
python -u scripts/tunesft.py \
    --dataset_name="data/einstein/einsteinsftheldout" \
    --output_dir="checkpoints/einsteinoptcorrect125m" \
    --save_steps=50 \
    --num_train_epochs=2 \
    --learning_rate=1e-4 \
    --model_name=facebook/opt-125m \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --warmup_steps=50 \
    --logging_steps=10

# EleutherAI/gpt-neo-125m 