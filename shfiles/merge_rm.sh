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
    --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bagofwords/nozero100k_tinysft_dpo/checkpoint-4000" \
    --base_model_name="models/bagofwords/tinybow_sft" \
    --output_name="models/bowdposfttiny"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bagofwords/nozero100k_1bnofa_rm/checkpoint-9000" \
#     --base_model_name="facebook/opt-1.3b" \
#     --output_name="models/rewards/bagofwords/1bnofarm"

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
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebow/50krmnp/best_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/revbow/revrm50k"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/reversebow/truncrmnp/best_checkpoint" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/revbow/revrmtrunc"

# export CUDA_VISIBLE_DEVICES=7
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="checkpoints/bagofwords/dpoplusbow50rm/step_100" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/bagofwords/50rmppo_s100_sft"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="checkpoints/bagofwords/dpoplusbow50rm/step_200" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/bagofwords/50rmppo_s200_sft"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/bagofwords/bowreversedata_reversebow_dpo/checkpoint-7000" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/bagofwords/tinybow_sft"

# export CUDA_VISIBLE_DEVICES=7
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="checkpoints/13rmultratrunc/checkpoint-10500" \
#     --base_model_name="allenai/tulu-2-13b" \
#     --output_name="models/rewards/13btruncrm"

# export CUDA_VISIBLE_DEVICES=7
# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/einsteinsftllama7b/checkpoint-10000" \
#     --base_model_name="/u/prasanns/research/rlhf-length-biases/models/llama" \
#     --output_name="models/sft7beinstein"

# python scripts/merge_peft_adapter.py \
#     --adapter_model_name="/u/prasanns/research/rlhf-length-biases/checkpoints/rmself25/checkpoint-7000" \
#     --base_model_name="facebook/opt-125m" \
#     --output_name="models/rewards/bowrmself25"



