# adapter for llama webgpt model
python merge_peft_adapter.py \
    --adapter_model_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/checkpoints/wgptllama/checkpoint-4000" \
    --base_model_name="/home/prasann/Projects/tfr-decoding/llama/llama" \
    --output_name="/home/prasann/Projects/rlhf-exploration/webgpt-llama/models/rewardmodel/"