# adapter for llama webgpt model
#python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/wgptllama/checkpoint-4000" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmodel/"

# adapter for steamSHP llama reward model
#python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/shp-llama/checkpoints/llamashp/checkpoint-19000" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/shp-llama/models/shprm/"

# adapter for SHP llama sft model
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/shp-llama/checkpoints/llamasft/checkpoint-2000" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/shp-llama/models/shpsft/"

# adapter for adv DA model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/checkpoints/rlcdsaved/checkpoint-3500" \
#     --base_model_name="/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/helprm7b/"

# # adapter for RLCD reward model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/advda/checkpoint-3500" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardadvda/"

export CUDA_VISIBLE_DEVICES=2,3
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/advtiebreak/checkpoint-4000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardadvtiebreak/"

# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/advmse/checkpoint-8000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardadvmse/"

# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/rm_da_randonly/checkpoint-4000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardrandaug/"

# new bigdata model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/checkpoints/rm_bigdata/checkpoint-16000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/sft" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/stack-llama/models/rewardbigdset/"

# webgpt with random DA
# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/webgptrandda/checkpoint-1500" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardrandda/"

# webgpt with TFRM (first attempt)
# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/tfrtest/checkpoint-4500" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardtfr"

# python merge_peft_adapter.py \
#    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/webgptmix/checkpoint-1500" \
#    --base_model_name="/home/prasann/Projects/tfr-decoding/apfarm_models/sft10k" \
#    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/mixrm"

# adapter for adv DA model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/checkpoints/rlcdsaved/checkpoint-2000" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/rlcd-llama/models/rlcdhelp/"

# adapter for adv DA model 
# python merge_peft_adapter.py \
#     --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/rmmix50/checkpoint-2500" \
#     --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
#     --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/mix50rm"

# APFarm human RM 
python merge_peft_adapter.py \
    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/apf/checkpoints/apfinithuman/checkpoint-1000" \
    --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
    --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/humanrm"

# APFarm GPT4 RM
python merge_peft_adapter.py \
    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/apf/checkpoints/apfgptrm/checkpoint-1500" \
    --base_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/sft10k" \
    --output_name="/home/prasann/Projects/rlhf-exploration/apf/models/gptrm"