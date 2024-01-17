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

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/80rmv2/_peft_last_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/expbow80"

# TODO look into trying out earlier checkpoints as well
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/30rmv2/_peft_last_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/expbow30"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/50rmv2/_peft_last_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/expbow50"

export CUDA_VISIBLE_DEVICES=7
python scripts/merge_peft_adapter.py \
    --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/rmdistrmix/_peft_last_checkpoint" \
    --base_model_name="facebook/opt-125m" \
    --output_name="models/rewards/distrmixrm"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/rmself25/checkpoint-7000" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/bowrmself25"



