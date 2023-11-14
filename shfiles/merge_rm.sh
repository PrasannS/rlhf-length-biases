python scripts/merge_peft_adapter.py \
    --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bigstack/checkpoint-3100" \
    --base_model_name="meta-llama/Llama-2-13b-hf" \
    --output_name="models/rewards/stackbigrm"

python scripts/merge_peft_adapter.py \
    --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bigrlcd/checkpoint-1800" \
    --base_model_name="meta-llama/Llama-2-13b-hf" \
    --output_name="models/rewards/rlcdbigrm"

python scripts/merge_peft_adapter.py \
    --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bigwgpt/checkpoint-1600" \
    --base_model_name="meta-llama/Llama-2-13b-hf" \
    --output_name="models/rewards/wgptbigrm"