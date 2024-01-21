export CUDA_VISIBLE_DEVICES=1
python -u datagen/tunesft.py \
    --output_dir="checkpoints/wikisft" \
    --save_steps=10000 \
    --num_train_epochs=10 \
    --learning_rate=1e-4