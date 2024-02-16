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

# export CUDA_VISIBLE_DEVICES=2
# python -u scripts/tunesft.py \
#     --dataset_name="data/einstein/einsteinsftheldout" \
#     --output_dir="checkpoints/einsteinoptcorrect125m" \
#     --save_steps=50 \
#     --num_train_epochs=2 \
#     --learning_rate=1e-4 \
#     --model_name=facebook/opt-125m \
#     --per_device_train_batch_size=16 \
#     --per_device_eval_batch_size=16 \
#     --warmup_steps=50 \
#     --logging_steps=10 

sftrun() {
    # NOTE that we need to feed things in a specific format    

    python -u scripts/tunesft.py \
        --dataset_name="data/${1}/${2}" \
        --output_dir="checkpoints/${1}/${2}_sft" \
        --save_steps=200 \
        --num_train_epochs=2 \
        --learning_rate=1e-5 \
        --model_name=${3} \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=16 \
        --warmup_steps=50 \
        --logging_steps=10
    
}

export CUDA_VISIBLE_DEVICES=2
sftrun "bagofwords" 'truncsftdata' "facebook/opt-125m"

# EleutherAI/gpt-neo-125m 