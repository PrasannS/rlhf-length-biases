# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/ultrafeed/ultrarmv1/checkpoint-5000" \
#     --base_model_name="/u/prasanns/research/rlhf-length-biases/models/llama" \
#     --output_name="models/rewards/ultrarm"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bigrlcd/checkpoint-1800" \
#     --base_model_name="meta-llama/Llama-2-13b-hf" \
#     --output_name="models/rewards/rlcdbigrm"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bigwgpt/checkpoint-1600" \
#     --base_model_name="meta-llama/Llama-2-13b-hf" \
#     --output_name="models/rewards/wgptbigrm"


# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/moresynth/bowrm/checkpoint-2400" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/minibowrm"

python scripts/merge_peft_adapter.py \
    --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/expbowrm/checkpoint-7800" \
    --base_model_name="facebook/opt-125m" \
    --output_name="models/rewards/expbowreward"

